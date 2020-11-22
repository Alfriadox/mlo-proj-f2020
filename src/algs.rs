
use rand_isaac::Isaac64Rng;
use rand::SeedableRng;
use std::ops::Mul;
use ndarray_rand::RandomExt;
use ndarray::{Array, Array1};
use ndarray_rand::rand_distr::{Bernoulli, StandardNormal};
use crate::{Graph, TriangleEstimate};
use sprs::{CsMatI, CsMat};
use std::ops::Div;
use nalgebra::{DMatrix, DVector, SymmetricEigen, Dynamic};


/// Use spectral counting to get the exact number of triangles in an undirected
/// graph.
///
/// `graph` should be a reference to the adjacency matrix of the graph.
///
/// Returns the number of triangles in the graph as an unsigned integer.
pub fn spectral_count(graph: &Graph) -> TriangleEstimate {
    // Convert the cell type from booleans to unsigned integers.
    let int_graph: CsMatI<u32, usize> = graph.map(|c| if *c {1} else {0});
    // Cube the graph's adjacency matrix.
    let cubed: CsMatI<u32, usize> = &(&int_graph * &int_graph) * &int_graph;

    // Take the diagonal sum using an iterator.
    let diag_sum: u32 = cubed.iter()
        // filter for diagonal cells only
        .filter(|(_, (r, c))| r == c)
        // take the sum
        .fold(0, |acc, (item, _)| acc+item);

    // return the sum of the diagonal divided by 6.
    return diag_sum/6;
}

/// The options available for generating random vectors in the TraceTriangle
/// algorithm.
pub enum RandomVector {
    /// Use Rademacher random vectors.
    Rademacher,
    /// Use normally generated random vectors.
    Normal
}

/// A structure to store the parameters and input used in the
/// TraceTriangle algorithm.
pub struct TraceTriangle {
    /// Which random vector generation method to use.
    pub random_vector_variant: RandomVector,
    /// The seed (if any) to seed the random number generator with.
    pub seed: Option<u64>,
    /// The gamma value specified in the algorithm.
    pub gamma: f64,
    /// The adjacency matrix of the undirected graph to operate on.
    pub graph: Graph,
}

impl TraceTriangle {
    /// Run the TraceTriangle algorithm using the parameters and input stored
    /// in `self`.
    ///
    /// # Panics
    /// This function will panic if:
    /// - The random number generator cannot be created
    ///     (system dependent, very unlikely).
    ///
    pub fn run(&self) -> TriangleEstimate {
        // Create random number generator. We use the Isaac64 generator
        // for its speed and quality.
        let mut rng: Isaac64Rng;
        // If a seed is available, use it.
        if let Some(seed) = self.seed {
            rng = Isaac64Rng::seed_from_u64(seed);
        } else {
            // Otherwise initialize from system entropy. This uses a system
            // reliant `getrandom` call. This is unlikely to panic, but may if
            // entropy cannot be securely provided.
            rng = Isaac64Rng::from_entropy();
        }

        // get the number of vertices in the graph.
        let n = self.graph.shape().0;
        // calculate M.
        let m = (n as f64)          // convert to 64-bit float for calculation
            .ln()                   // take the natural log
            .powf(2f64)          // square it
            .mul(&self.gamma) // multiply it by gamma
            .ceil() as u64;             // take the ceiling and convert back to integer.

        // iterate over [0, M-1]
        return (0..m)
            .map(|_| {
                // Make the x vector.
                let x: Array1<f64> = match self.random_vector_variant {
                    // Using the Rademacher method.
                    RandomVector::Rademacher => {
                        // make the bernoulli distribution
                        let dist = Bernoulli::new(0.5).unwrap();
                        // sample from the distribution into an array
                        Array::random_using(n, dist, &mut rng)
                            // and then map bernoulli successes and failures to
                            // +1 and -1 respectively
                            .map(|bern: &bool| if *bern {1f64} else {-1f64})
                    },
                    // Using the normal method.
                    RandomVector::Normal => {
                        Array::random_using(n, StandardNormal, &mut rng)
                    }
                };

                // map the graph from booleans to floats for multiplication.
                let mapped: CsMat<f64> = self.graph.map(|c| if *c {1f64} else {0f64});
                // multiply the graph by array x. This corresponds to y = Ax.
                let y: Array1<f64> = &mapped * &x;
                // dot y with the multiplication of the graph by y.
                // this corresponds to (T_i = y^T A y / 6)
                y.dot(&(&mapped * &y)) / 6f64
            })
            // take the sum
            .fold(0f64, |acc, it| acc+it)
            // divide by M to get the average
            .div(m as f64)
            // and then convert to an integer
            .floor() as TriangleEstimate;
    }
}

/// A structure to store parameters and input for the EigenTriangle algorithm.
pub struct EigenTriangle {
    /// The random number generator seed (if any).
    pub seed: Option<u64>,
    /// The tolerance passed to the EigenTriangle algorithm.
    pub tolerance: f64,
    /// The adjacency matrix of the graph to operate on.
    pub graph: Graph,
}

impl EigenTriangle {
    /// Lanczos algorithm implementation.
    ///
    /// Returns tuple containing `(v_j+1, alpha_j, beta_j+1)`.
    ///
    /// May return `None` if an invalid/bad condition is encountered.
    /// This indicates the need to restart the EigenTriangle algorithm.
    fn lanczos(
        // Reference to the input parameters passed to the parent
        // EigenTriangle algorithm.
        &self,
        // Reference to the adjacency matrix of the graph converted from
        // booleans to unsigned bytes to operate on.
        a: &CsMat<f64>,
        // Reference to v_j-1.
        vj_prev: &Array1<f64>,
        // Reference to v_j.
        vj: &Array1<f64>,
        // Copy of beta_j value.
        beta_j: f64,
        // Mutable reference to the random number generator.
        rng: &mut Isaac64Rng
    ) -> Option<(Array1<f64>, f64, f64)> {
        // Get the number of vertices, N.
        let n = a.shape().0;

        // Calculate w_prime_j.
        let w_prime_j: Array1<f64> = a * vj - beta_j*vj_prev;

        // Calculate alpha_j as defined in the algorithm spec.
        // This inner product is calculated using a dot product since
        // both arguments are euclidean vectors.
        let alpha_j: f64 = w_prime_j.dot(vj);

        // Calculate w_j as defined in the algorithm spec.
        let wj: Array1<f64> = w_prime_j - alpha_j*vj;

        // If w_j is zero (or sufficiently close), we need to restart.
        if wj.iter().all(|ev| ev.abs() <= f64::EPSILON) {
            return None;
        }

        // Calculate beta_j+1 as defined in the algorithm spec.
        // Take the euclidean norm of w_j by dotting it with itself and
        // taking the square root of the result.
        let beta_j_next: f64 = wj.dot(&wj).sqrt();

        // Calculate v_j+1 as defined in the algorithm spec.
        let vj_next: Array1<f64>;
        // Check if beta_j+1 is equal to zero. Since floating point equality
        // can be finicky, we check if beta_j+1 is within an extremely small
        // margin of zero.
        if beta_j_next.abs() <= f64::EPSILON {
            // If beta_j+1 is zero (or sufficiently close), select a random
            // vector for v_j+1 using the same method used to select v_1.
            // Generate a random vector.
            let rand_vec: Array1<f64> = Array::random_using(n, StandardNormal, rng);
            // Calculate euclidean norm of the random vector.
            let norm: f64 = rand_vec.dot(&rand_vec).sqrt();
            // Divide random vector by euclidean norm to get normalized vector.
            vj_next = rand_vec / norm;
        } else {
            // If beta_j+1 is not zero, calculate v_j+1 as defined in the
            // algorithm spec.
            vj_next = wj / beta_j_next;
        }

        // Return the calculated values.
        return Some((vj_next, alpha_j, beta_j_next));
    }

    /// Function used to run the EigenTriangle algorithm on a set of inputs.
    pub fn run(&self) -> TriangleEstimate {
        // Create a random number generator.
        let mut rng: Isaac64Rng;

        // Seed the random number generator.
        if let Some(seed) = self.seed {
            rng = Isaac64Rng::seed_from_u64(seed);
        } else {
            // This may panic if entropy cannot be securely accessed from the
            // system.
            rng = Isaac64Rng::from_entropy();
        }

        // Convert the adjacency matrix of the graph to 64-bit floats to make
        // it easier to operate on.
        let a: CsMat<f64> = self.graph.map(|c| if *c {1f64} else {0f64});
        // Get the number of vertices in the graph.
        let n: usize = a.shape().0;

        // Seed v0 as zeros and v1 as a random normalized vector as defined in
        // the algorithm.
        let v0: Array1<f64> = Array::zeros(n);
        // Create a random vector.
        let rand_vec: Array1<f64> = Array::random_using(n, StandardNormal, &mut rng);
        // Calculate euclidean norm of the random vector.
        let norm: f64 = rand_vec.dot(&rand_vec).sqrt();
        // Divide random vector by euclidean norm to get normalized vector, v_1.
        let v1: Array1<f64> = rand_vec / norm;

        // Seed beta_1 as 0 as defined in the algorithm.
        let beta_1: f64 = 0f64;

        // Create j, the iterating variable through k as defined
        // in the algorithm spec.
        let mut j: usize = 1;

        // Build T, a tri-diagonal symmetric matrix, as defined in the
        // algorithm spec. First create an empty matrix of size KxK.
        let mut t: DMatrix<f64> = DMatrix::zeros(j, j);

        // Calculate the first values to put in the matrix T using Lanczos
        // method as defined in the algorithm spec.
        let lanczos_1: Option<(Array1<f64>, f64, f64)> =
            self.lanczos(&a, &v0, &v1, beta_1, &mut rng);
        // Check if lanczos failed, restart if so.
        if lanczos_1.is_none() {
            // Restart.
            return self.run();
        }
        // We did not fail, so we should unwrap the results, and proceed.
        let (v2, alpha_1, beta_2) = lanczos_1.unwrap();

        // Set alpha_1 in T.
        t[(0,0)] = alpha_1;

        // Create reference variables to point to the parameters to pass
        // to Lanczos method.
        let mut vj_next: Array1<f64> = v2;
        let mut vj: Array1<f64> = v1;
        let mut beta_j_next: f64 = beta_2;

        // Assume k > 2. Iterate j up to 2.
        while j < 2 {
            // Increment j.
            j += 1;

            // Enlarge T.
            t = t.resize(j, j, 0f64);

            // Add beta_j (previously beta_j+1) to T.
            t[(j-1, j-2)] = beta_j_next;
            t[(j-2, j-1)] = beta_j_next;

            // Calculate new values from Lanczos method.
            let lanczos_j: Option<(Array1<f64>, f64, f64)> =
                self.lanczos(&a, &vj, &vj_next, beta_j_next, &mut rng);
            // Check that Lanczos method didn't fail.
            if lanczos_j.is_none() {
                // If it did, restart.
                return self.run();
            }
            // Otherwise update T, and the reference variables.
            let (new_vj_next, alpha_j, new_beta_j_next) = lanczos_j.unwrap();
            t[(j-1, j-1)] = alpha_j;
            vj = vj_next;
            vj_next = new_vj_next;
            beta_j_next = new_beta_j_next;
        }

        // Create a variable to store the current tolerance of the algorithm.
        let mut current_tol: f64;

        // Iterate until within tolerances.
        loop {
            // Debugging code.
            eprintln!("Iteration {}:\n T: {}", j, t);

            // Calculate eigenvalues.
            let eigen_decomp: SymmetricEigen<f64, Dynamic> = t.symmetric_eigen();
            let eigen_vals: &DVector<f64> = &eigen_decomp.eigenvalues;

            eprintln!("Eigenvalues: {:?}", eigen_vals);

            // Calculate the current tolerance value.
            // Get the newest eigenvalue.
            let lambda_j: f64 = eigen_vals[j-1];
            // Sum the cubes of all eigenvalues.
            let sum_cubed_eigenvalues: f64 = eigen_vals
                // Cube eigenvalues.
                .map(|ev| ev.powf(3f64))
                // Sum cubed eigenvalues.
                .fold(0f64, |acc, ev3| acc + ev3);
            // Divide the absolute value of the cube of lambda_j by
            // the sum of the cubes of current eigenvalues.
            current_tol = lambda_j.powf(3f64).abs() / sum_cubed_eigenvalues;

            eprintln!("Calculated tolerance: {}", current_tol);

            // Check if we are within tolerances.
            if 0f64 <= current_tol && current_tol <= self.tolerance {
                // Return 1/6 of the sum of the cubes of the eigenvalues.
                let res: f64 = (1f64/6f64) * eigen_vals
                    // Cube each eigen value.
                    .map(|ev| ev.powf(3f64))
                    // Take the sum of the cubed eigenvalues.
                    .fold(0f64, |acc, ev3| acc + ev3);

                return res as TriangleEstimate;
            } else {
                // If not within tolerances, we need to iterate further.
                // Increment j.
                j += 1;

                // Rebuild and enlarge T.
                t = eigen_decomp.recompose();
                t = t.resize(j, j, 0f64);

                // Add beta_j (previously beta_j+1) to T.
                t[(j-1, j-2)] = beta_j_next;
                t[(j-2, j-1)] = beta_j_next;

                // Calculate new values from Lanczos method.
                let lanczos_j: Option<(Array1<f64>, f64, f64)> =
                    self.lanczos(&a, &vj, &vj_next, beta_j_next, &mut rng);
                // Check that Lanczos method didn't fail.
                if lanczos_j.is_none() {
                    // If it did, restart.
                    return self.run();
                }
                // Otherwise update T, and the reference variables.
                let (new_vj_next, alpha_j, new_beta_j_next) = lanczos_j.unwrap();
                t[(j-1, j-1)] = alpha_j;
                vj = vj_next;
                vj_next = new_vj_next;
                beta_j_next = new_beta_j_next;
            }
        }
    }
}
