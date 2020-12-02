use std::time::{Instant, Duration};
use crate::TriangleEstimate;
use crate::TRIALS;
use crate::boilerplate::Dataset;

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
#[derive(Clone, Debug)]
pub struct AlgorithmBenchmark {
    /// The dataset this is running on.
    dataset: &'static Dataset,
    /// The name of this algorithm.
    alg_name: &'static str,
    /// A list of the time and output from each trial.
    inner: Vec<(Duration, TriangleEstimate)>
}

/// Serializable benchmark record that can be passed to the CSV writer.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BenchmarkRecord {
    /// The name of the algorithm.
    alg_name: &'static str,
    /// The trial number of this algorithm on this dataset.
    trial_num: String,
    /// The amount of time it took to run the algorithm in microseconds.
    runtime: u128,
    /// The returned result of the algorithm.
    result: TriangleEstimate,
    /// The gamma value passed to TriangleTrace alg.
    gamma: Option<f64>,
    /// The iterations parameter passed to the EigenTriangle alg.
    max_iters: Option<usize>,
    /// The path of the dataset that the algorithm was run on.
    ds_path: &'static str,
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
        ds: &'static Dataset,
        alg_name: &'static str,
        alg: fn(&I) -> TriangleEstimate,
        input: &I,
    ) -> Self
    where I: Sync
    {
        // For each trial
        let data = (0..TRIALS)
            .into_iter()
            // run the algorithm for each trial, collecting results.
            .map(|tn| {
                let r = time(alg, input);
                println!("{}: Finished trial {} of {}. ({})",
                         ds.path, tn+1, TRIALS, alg_name);
                r
            })
            // collect into a list
            .collect();

        Self {
            dataset: ds,
            alg_name,
            inner: data
        }
    }

    /// Convert this benchmark into a list of serializable records that can be saved to
    /// the CSV file.
    pub fn to_records(
        &self,
        gamma: Option<f64>,
        max_iters: Option<usize>
    ) -> Vec<BenchmarkRecord> {
        // Iterate over the stored results.
        self.inner
            .iter()
            // Enumerate the iteration.
            .enumerate()
            // Convert to BenchmarkRecords.
            .map(|(trial_num, (runtime, result))| {
                BenchmarkRecord {
                    alg_name: self.alg_name,
                    ds_path: self.dataset.path,
                    trial_num: format!("{}", trial_num + 1),
                    runtime: runtime.as_micros(),
                    result: *result,
                    gamma,
                    max_iters
                }
            })
            // Collect into the output.
            .collect()
    }
}

