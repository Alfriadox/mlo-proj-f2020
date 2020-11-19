use std::time::{Instant, Duration};
use crate::TriangleEstimate;
use crate::boilerplate::Dataset;
use crate::TRIALS;
use rayon::prelude::*;
use std::io::Write;

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
    /// `alg_name` should be a string describing the algorithm's name. This is
    /// only used for printing status messages.
    pub fn run<I>(
        alg_name: String,
        alg: fn(&I) -> TriangleEstimate,
        input: &I,
    ) -> Self
    where I: Sync
    {
        // for each trial (in parallel)
        let data = (0..TRIALS)
            .map(|trial_num| {
                println!("[{}] Running trial {:03} of {:03}...", alg_name, trial_num+1, TRIALS);
                // run the algorithm and measure runtime
                let result = time(alg, input);

                println!("[{}] Trial {} of {} complete in {} ms. (returned {})",
                         alg_name, trial_num+1, TRIALS, result.0.as_millis(), result.1);

                // return the produced result
                result
            })
            // collect into a list
            .collect();

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