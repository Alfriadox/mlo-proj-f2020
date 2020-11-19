use std::time::{Duration, Instant};
use rand_isaac::Isaac64Rng;
use rand::SeedableRng;
use std::ops::Mul;
use rayon::prelude::*;
use ndarray_rand::RandomExt;
use ndarray::{Array, Array1};
use ndarray_rand::rand_distr::{Bernoulli, StandardNormal};
use crate::Graph;

/// Function to measure the runtime of an algorithm on a given input.
///
/// This function will panic if the algorithm panics.
///
/// This function returns a tuple of the runtime and algorithm output as
/// (duration, output).
pub fn time<F, I, O>(alg: F, input: I) -> (Duration, O)
where F: FnOnce(I) -> O {
    let start_time = Instant::now();
    let result = alg(input);
    let elapsed = start_time.elapsed();
    return (elapsed, result);
}


/// Use spectral counting to get the exact number of triangles in an undirected
/// graph.
///
/// `graph` should be a reference to the adjacency matrix of the graph.
///
/// Returns the number of triangles in the graph as an unsigned integer.
pub fn spectral_count(graph: &Graph) -> f64 {
    // Cube the graph's adjacency matrix.
    let cubed: Graph = &(graph * graph) * graph;

    // Take the diagonal sum using an iterator.
    let diag_sum = cubed.iter()
        // filter for diagonal cells only
        .filter(|(_, (r, c))| r == c)
        // take the sum
        .fold(0f64, |acc, (item, _)| acc+item);

    // return the sum of the diagonal divided by 6.
    return diag_sum/6f64;
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
///
/// This structure is parameterized over lifetime `'graph` which
/// references the lifetime for which the input matrix is valid.
pub struct TraceTriangle<'graph> {
    /// Which random vector generation method to use.
    pub random_vector_variant: RandomVector,
    /// The seed (if any) to seed the random number generator with.
    pub seed: Option<u64>,
    /// The gamma value specified in the algorithm.
    pub gamma: f64,
    /// A reference (with lifetime `'graph`) to the adjacency matrix of the
    /// undirected graph to operate on.
    pub graph: &'graph Graph,
}

impl<'graph> TraceTriangle<'graph> {
    /// Run the TraceTriangle algorithm using the parameters and input stored
    /// in `self`.
    ///
    /// # Panics
    /// This function will panic if:
    /// - The random number generator cannot be created
    ///     (system dependent, very unlikely).
    ///
    pub fn run(&self) -> f64 {
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
            .powf(2.)            // square it
            .mul(&self.gamma) // multiply it by gamma
            .ceil() as u64;              // take the ceiling and convert back to integer.

        // iterate over [0, M-1] in parallel using rayon
        return (0..m).into_par_iter()
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

                // multiply the graph by array x. This corresponds to y = Ax.
                let y: Array1<f64> = self.graph * &(x.t());
            });
    }
}