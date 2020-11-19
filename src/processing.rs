use std::time::{Instant, Duration};
use crate::TriangleEstimate;
use crate::boilerplate::Dataset;
use crate::TRIALS;
use rayon::prelude::*;
use std::io::Write;
use indicatif::{MultiProgress, ProgressStyle, ProgressBar};
use std::sync::Arc;

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

/// A structure to store the output and runtime of an algorithm over a
/// number of trials.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct AlgorithmBenchmark {
    /// A list of the time and output from each trial.
    inner: Vec<(Duration, TriangleEstimate)>
}

// F is the algorithm function type, I is the input type.
impl AlgorithmBenchmark {
    /// Benchmark an algorithm over a specific input in parallel and store
    /// the results into a AlgorithmBenchmark object. The number of trials
    /// is determined by the `TRIALS` constant in `main.rs`.
    ///
    /// This method is parameterized over `I`, the input type passed to the
    /// algorithm.
    ///
    /// `prefix` should be a string describing the algorithm's name and the
    /// dataset. This is only used for printing the progress bar.
    pub fn run<I>(
        progress_bar: ProgressBar,
        alg: fn(&I) -> TriangleEstimate,
        input: &I,
    ) -> Self
    where I: Sync
    {
        // for each trial
        let data = progress_bar.wrap_iter(0..TRIALS)
            // run the algorithm for each trial, collecting results.
            .map(|_| time(alg, input))
            // collect into a list
            .collect();

        progress_bar.finish();

        Self { inner: data }
    }

    /// Get the mean duration and estimated triangle count.
    pub fn mean(&self) -> (Duration, TriangleEstimate) {
        let n = self.inner.len() as u32;
        let sum = self.inner.iter()
            .fold(
                (Duration::from_secs(0), 0),
                |acc, it| (acc.0 + it.0, acc.1 + it.1));
        (sum.0 / n , sum.1 / n)
    }
}