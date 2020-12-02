
use rand_isaac::Isaac64Rng;
use rand::SeedableRng;
use std::ops::Mul;
use ndarray_rand::RandomExt;
use ndarray::{Array, Array1, Array2};
use ndarray_rand::rand_distr::{Bernoulli, StandardNormal};
use crate::{Graph, TriangleEstimate};
use sprs::{CsMatI, CsMat};
use std::ops::Div;
use nalgebra::{DMatrix, DVector, DMatrixSlice, DVectorSlice};
use eigenvalues::algorithms::lanczos::HermitianLanczos;
use eigenvalues::SpectrumTarget;
use eigenvalues::matrix_operations::MatrixOperations;


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
    /// The tolerance passed to the EigenTriangle algorithm.
    pub maximum_iterations: usize,
    /// The adjacency matrix of the graph to operate on.
    pub graph: Graph,
}

/// Wrapper around Graph to implement the Hermetian Lanczos algorithm.
#[derive(Clone)]
pub struct HLGraph(CsMat<f64>);

impl MatrixOperations for HLGraph {
    fn matrix_vector_prod(&self, vs: DVectorSlice<'_, f64>) -> DVector<f64> {
        // Convert the vector to an array1 to multiply against matrix.
        let c: Array1<f64> = vs.iter().map(|f| *f).collect();

        // Multiply the matrix by the vector.
        let multiplied: Array1<f64> = &self.0 * &c;

        // Convert the result of the multiplication to a DVector.
        let r: Vec<f64> = multiplied.iter().map(|f| *f).collect();
        DVector::from_vec(r)
    }

    fn matrix_matrix_prod(&self, mtx: DMatrixSlice<'_, f64>) -> DMatrix<f64> {
        // Convert the matrix into an Array2 for multiplication.
        let c: Array2<f64> = Array2::from_shape_fn(
            mtx.shape(),
            |(r, c)| mtx[(r, c)]
        );

        // Multiply the matrix by the graph.
        let multiplied: Array2<f64> = &self.0 * &c;

        // Convert the multiplication result to a DMatrix and return.
        let rvec: Vec<f64> = multiplied.iter().map(|f| *f).collect();
        DMatrix::from_vec(multiplied.nrows(), multiplied.ncols(), rvec)
    }

    fn diagonal(&self) -> DVector<f64> {
        // Filter and collect the diagonal cells.
        let diag_vec: Vec<f64> = self.0.iter()
            .filter(|(_, (row, col))| {
                row == col
            })
            .map(|(v, _)| *v)
            .collect();
        // Return the vector.
        DVector::from_vec(diag_vec)
    }

    fn set_diagonal(&mut self, diag: &DVector<f64>) {
        diag
            .iter()
            .enumerate()
            .for_each(|(rc, v)| {
                self.0[[rc, rc]] = *v;
            })
    }

    fn ncols(&self) -> usize {
        self.0.cols()
    }

    fn nrows(&self) -> usize {
        self.0.rows()
    }
}

impl EigenTriangle {
    /// Implementation of Lanczos method ported from https://docs.rs/eigenvalues/0.3.1/src/eigenvalues/algorithms/lanczos.rs.html.
    /// And adapted for sparse matrices.
    ///
    /// This function takes a reference to self and uses the
    /// adjacency matrix of the graph and the max_iterations parameter.
    ///
    /// This function returns a list of eigenvalues.
    fn hermetian_lanczos(&self) -> DVector<f64> {
        // Define the tolerance of this algorithm to be 1x10^-8.
        let tolerance: f64 = 1E-8;

        // Make vectors to store diagonal and off diagonal elements.
        let alphas: Array1<f64> = Array1::zeros(self.maximum_iterations);
        let betas: Array1<f64> = Array1::zeros(self.maximum_iterations - 1);

        // Matrix with orthogonal vectors.
        let mut vs: Array2<f64> = Array2::zeros(
            (self.graph.rows(), self.maximum_iterations)
        );

        // First vector for v (random vector of euclidean length 1).
        let rand_vec: Array1<f64> = Array::random(self.graph.rows(), &StandardNormal);
        // Normalize the random vector.
        let norm: f64 = rand_vec.dot(&rand_vec).sqrt();
        let xs: Array1<f64> = rand_vec / norm;
        // Add the normalized random vector.
        unimplemented!()
    }

    /// Construct a tri-diagonal matrix of the alphas and betas
    /// for the lanczos method.
    fn construct_tridiagonal(alphas: &Array1<f64>, betas: &Array1<f64>) -> DMatrix<f64> {
        // Get the length of the alpha vector. This will be the dimension
        // of the returned matrix.
        let dim: usize = alphas.len();
        // Make a lambda function and use it to populate and return the
        // tri-diagonal matrix.
        let lambda = |i: usize, j: usize| {
            if i == j {
                alphas[i]
            } else if i == j + 1 {
                betas[j]
            } else if j == i + 1 {
                betas[i]
            } else {
                0f64
            }
        };
        DMatrix::<f64>::from_fn(dim, dim, lambda)
    }

    /// Function used to run the EigenTriangle algorithm on a set of inputs.
    pub fn run(&self) -> TriangleEstimate {
        // Get the number of vertices in the adjacency matrix of the graph.
        let n: usize = self.graph.shape().0;
        // Convert the adjacency matrix of the graph to a dense 64-bit float
        // matrix to pass it to the Lanczos implementation.
        let a: DMatrix<f64> =
            DMatrix::from_fn(
                n,
                n,
                |row, col| self.graph
                    .get(row, col)
                    .map(|e| if *e {1f64} else {0f64})
                    .unwrap_or(0f64));

        // Execute lanczos algorithm to get the eigenvalues of the adjacency
        // matrix.
        let lanczos: HermitianLanczos = HermitianLanczos::new(
            a,
            self.maximum_iterations,
            SpectrumTarget::Highest
        ).unwrap();

        // Extract the eigenvalues from lanczos method.
        let eigen_values: DVector<f64> = lanczos.eigenvalues;

        // Return 1/6 of the sum of the cubes of the eigenvalues.
        let res: f64 = (1f64/6f64) * eigen_values
            // Cube each eigen value.
            .map(|ev| ev.powf(3f64))
            // Take the sum of the cubed eigenvalues.
            .fold(0f64, |acc, ev3| acc + ev3);

        return res as TriangleEstimate;
    }
}
