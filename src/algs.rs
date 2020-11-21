
use rand_isaac::Isaac64Rng;
use rand::SeedableRng;
use std::ops::Mul;
use ndarray_rand::RandomExt;
use ndarray::{Array, Array1};
use ndarray_rand::rand_distr::{Bernoulli, StandardNormal};
use crate::{Graph, TriangleEstimate};
use sprs::{CsMatI, CsMat};
use std::ops::Div;
use nalgebra::{DMatrix};


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
    /// This function take a reference to the input parameters of the
    /// EigenTriangle algorithm form of a reference to self and an
    /// unsigned integer, k.
    ///
    /// This method may panic if the random number generator panics.
    ///
    /// This method should return k eigenvalues as 64-bit floats.
    fn lanczos(&self, k: usize) -> Vec<f64> {
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

        // Get the number of vertices, N in the graph.
        let n: usize = self.graph.shape().0;

        // Get the sparse matrix A, the adjacency matrix of the graph
        // converted from booleans to floating point values to operate on.
        let a: CsMat<f64> = self.graph.map(|e| if *e { 1f64 } else { -1f64 });

        // Make a list for intermediate vectors V.
        // Vectors in V are dense vectors of length n.
        let mut v: Vec<Array1<f64>> = Vec::with_capacity(k+2);
        // v0 is defined as the zero vector
        v.push(Array::zeros(n));
        // v1 is defined as an arbitrarily vector of euclidean norm 1.
        let v1: Array1<f64> = Array::random_using(n, StandardNormal, &mut rng);
        // Normalize v1 so that the euclidean norm is 1.
        // Get the euclidean norm by taking the square root of the dot product
        // of v with itself.
        let v1_norm: f64 = v1.dot(&v1).sqrt();
        // Divide v by it's norm to get a scaled v1 with euclidean norm 1.
        // Add the scaled v1 to the list V.
        v.push(v1/v1_norm);

        // List of intermediate beta values. This is indexed from 1
        // (so subtract 1 when indexing into this list).
        let mut beta: Vec<f64> = Vec::with_capacity(k+1);
        // Beta 1 is defined as 0.
        beta.push(0f64);

        // List of intermediate alpha values. This is also indexed from 1
        // (so subtract 1 when indexing into this list).
        let mut alpha: Vec<f64> = Vec::with_capacity(k+1);

        // Iterate j from 1 to k (inclusive).
        for j in 1..(k+1) {
            // Calculate w_prime_j as defined in the algorithm spec.
            // First get v_j, v_j-1, and beta_j.
            // These accesses will panic if out of bounds.
            let vj: &Array1<f64> = v.get(j).unwrap();
            let vj_prev: &Array1<f64> = v.get(j-1).unwrap();
            let beta_j: f64 = beta[j-1];
            // Calculate w_prime_j.
            let w_prime_j: Array1<f64> = (&a * vj) - beta_j*vj_prev;

            // Calculate alpha_j as defined in the algorithm spec.
            // This inner product is calculated using a dot product since
            // both arguments are euclidean vectors.
            let alpha_j: f64 = w_prime_j.dot(vj);
            // Add the calculated alpha to the intermediate alpha list.
            alpha.push(alpha_j);

            // Calculate w_j as defined in the algorithm spec.
            let wj: Array1<f64> = w_prime_j - alpha_j*vj;

            // Calculate beta_j+1 as defined in the algorithm spec.
            // Take the euclidean norm of w_j by dotting it with itself and
            // taking the square root of the result.
            let beta_j_next: f64 = wj.dot(&wj).sqrt();
            // Add the calculated beta to the list of betas.
            beta.push(beta_j_next);

            // Calculate v_j+1 as defined in the algorithm spec.
            let vj_next: Array1<f64>;
            // Check if beta_j+1 is equal to zero. Since floating point equality
            // can be finicky, we check if beta_j+1 is within an extremely small
            // margin of zero.
            if beta_j_next.abs() <= f64::EPSILON {
                // If beta_j+1 is zero (or sufficiently close), select a random
                // vector for v_j+1 using the same method used to select v_1.
                // Generate a random vector.
                let rand_vec: Array1<f64> = Array::random_using(n, StandardNormal, &mut rng);
                // Calculate euclidean norm of the random vector.
                let norm: f64 = rand_vec.dot(&rand_vec).sqrt();
                // Divide random vector by euclidean norm to get normalized vector.
                vj_next = rand_vec / norm;
            } else {
                // If beta_j+1 is not zero, calculate v_j+1 as defined in the
                // algorithm spec.
                vj_next = wj / beta_j_next;
            }
            // Add v_j+1 to the list of vectors V.
            v.push(vj_next);
        }

        // Once all the vectors in V and all of the alpha and beta values have
        // been calculated, build T, a tri-diagonal symmetric matrix, as
        // defined in the algorithm spec.
        // First create an empty matrix of size KxK.
        let mut t: DMatrix<f64> = DMatrix::zeros(k, k);

        // Next, add the diagonals to the matrix.
        // We only add the center and lower diagonals as that is all that is
        // required to do a symmetric eigen-decomposition.
        // The center diagonal is set here using an iterator.
        t.set_partial_diagonal(alpha.into_iter());

        // Then the off-diagonal is set manually. We skip the first beta value,
        // since the matrix T contains only beta_2 and on.
        // Create iterator over ((row, col), beta_val) items, where (row, col)
        // is the index into the matrix T.
        let idx_beta_iter = (1..)
            .zip(0..)
            .zip(beta.into_iter().skip(1));
        // Consume the iterator, setting the lower off-diagonal of T with
        // beta_2 through beta_k.
        for (index, beta_val) in idx_beta_iter {
            t[index] = beta_val;
        }

        // Finally, calculate and return the eigenvalues of T.
        t.symmetric_eigen()
            .eigenvalues
            .iter()
            .map(|ptr: &f64| *ptr)
            .collect()
    }

    /// Function used to run the EigenTriangle algorithm on a set of inputs.
    pub fn run(&self) -> TriangleEstimate {
        // Create a list of the returned eigen values from Lanczos method.
        let mut lambda: Vec<Vec<f64>> = Vec::new();
        // Get the first eigenvalues at k = 1. Add this to the list of returned
        // eigenvalues.
        let lambda_1: Vec<f64> = self.lanczos(1);
        lambda.push(lambda_1);

        // Create a mutable loop variable, i, as an unsigned integer.
        let mut i: usize = 1;

        unimplemented!()
    }
}
