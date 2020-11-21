use crate::boilerplate::Dataset;
use sprs::CsMat;
use crate::processing::{AlgorithmBenchmark, BenchmarkRecord};
use crate::algs::{spectral_count, TraceTriangle, RandomVector};
use indicatif::{MultiProgress, ProgressStyle, ProgressBar};
use std::thread;
use std::fs;
use csv::Writer;
use std::thread::JoinHandle;
use std::fs::File;

#[macro_use]
extern crate serde;


/// The boilerplate module contains boilerplate code for reading and
/// parsing graphs from the file system.
mod boilerplate;

/// The algs module contains implementations of the triangle counting and
/// approximation algorithms.
mod algs;

/// The processing module contains functions and structures for timing and
/// processing the results of the test cases.
mod processing;

/// The number of trials of every algorithm to run on each dataset.
pub const TRIALS: u8 = 10;

/// A constant array representing the datasets to test.
/// Add or remove entries as necessary.
///
/// See the `boilerplate` module for structure documentation.
const DATASETS: &'static [Dataset] = &[
    Dataset {
        path: "data/twitch/ES/musae_ES_edges.csv",
        nodes: 4_648,
        csv_delimiter: b',',
        has_header_row: true,
        comment_char: None,
    },
    Dataset {
        path: "data/twitch/ENGB/musae_ENGB_edges.csv",
        nodes: 7_126,
        csv_delimiter: b',',
        has_header_row: true,
        comment_char: None
    },
    Dataset {
        path: "data/twitch/DE/musae_DE_edges.csv",
        nodes: 9_498,
        csv_delimiter: b',',
        has_header_row: true,
        comment_char: None
    },
    Dataset {
        path: "data/twitch/FR/musae_FR_edges.csv",
        nodes: 6_549,
        csv_delimiter: b',',
        has_header_row: true,
        comment_char: None
    },
    Dataset {
        path: "data/twitch/PTBR/musae_PTBR_edges.csv",
        nodes: 1_912,
        csv_delimiter: b',',
        has_header_row: true,
        comment_char: None
    },
    Dataset {
        path: "data/twitch/RU/musae_RU_edges.csv",
        nodes: 4_385,
        csv_delimiter: b',',
        has_header_row: true,
        comment_char: None
    },
    Dataset {
        path: "data/wikipedia/chameleon/musae_chameleon_edges.csv",
        nodes: 2_277,
        csv_delimiter: b',',
        has_header_row: true,
        comment_char: None
    },
    Dataset {
        path: "data/wikipedia/crocodile/musae_crocodile_edges.csv",
        nodes: 11_631,
        csv_delimiter: b',',
        has_header_row: true,
        comment_char: None,
    },
    Dataset {
        path: "data/wikipedia/squirrel/musae_squirrel_edges.csv",
        nodes: 5_201,
        csv_delimiter: b',',
        has_header_row: true,
        comment_char: None,
    },
];

/// The type used to represent graphs. This is currently an adjacency matrix
/// with boolean cells.
pub type Graph = CsMat<bool>;

/// The type used to represent triangle counts (or estimates). This is currently
/// a 32-bit unsigned integer.
pub type TriangleEstimate = u32;

fn main() {
    // create thread aware progressbar manager.
    let multibar = MultiProgress::new();

    // specify the styling on algorithm testing progress bars.
    let algs_style = ProgressStyle::default_bar()
        .template("{spinner:.blue} {prefix:80} [{elapsed_precise}] {bar:50.green} [{eta_precise}] TRIAL {pos:>}/{len}");

    let read_style = ProgressStyle::default_bar()
        .template("{spinner:.blue} {prefix:80} [{elapsed_precise}] {bar:50.cyan} [{eta_precise}] {bytes:>}/{total_bytes} ({bytes_per_sec}) ");

    // utility closure to create a progress bar for a given dataset and algorithm.
    let make_alg_bar = |ds: &Dataset, alg_name: &str| -> ProgressBar {
        // create a new progress bar on the parent multibar.
        let bar = multibar.add(ProgressBar::new(TRIALS as u64))
            // with the global style
            .with_style(algs_style.clone());

        // set the bar's prefix to reference the dataset and algorithm name.
        bar.set_prefix(format!("{} ({}) (N: {})", ds.path, alg_name, ds.nodes).as_str());

        // set the bar to automatically tick every 300 ms.
        bar.enable_steady_tick(200);

        // return the bar.
        bar
    };

    // utility closure to create a progressbar for a given CSV file read.
    let make_fs_bar = |ds: &Dataset| -> ProgressBar {
        // get the file size.
        let file_size = fs::metadata(ds.path)
            .expect("Could not get file metadata.")
            .len();

        // create a new progress bar.
        let bar = multibar.add(ProgressBar::new(file_size))
            .with_style(read_style.clone());

        // set the bar's prefix to refer to the dataset path.
        bar.set_prefix(format!("READ {} (N: {})", ds.path, ds.nodes).as_str());

        // set the bar to automatically tick every 300 ms.
        bar.enable_steady_tick(200);

        // return the created bar.
        bar
    };

    // Create a list of thread handles to receive results from on completion.
    let mut threads: Vec<JoinHandle<Vec<BenchmarkRecord>>> = Vec::with_capacity(DATASETS.len());

    for dataset in DATASETS {
        // make a progress bar for the file reading
        let io_bar = make_fs_bar(dataset);
        // make a status bar for the spectral count.
        let spectral_count_bar = make_alg_bar(dataset, "Spectral Count");
        // make bar for trace_triangles using the Rademacher method.
        let ttr_bar = make_alg_bar(dataset, "TraceTrianglesR");
        // make bar for trace_triangles using the Normal/ gaussian method.
        let ttn_bar = make_alg_bar(dataset, "TraceTrianglesN");

        // Spawn a child thread to operate on the dataset.
        let join_handle: JoinHandle<Vec<BenchmarkRecord>> = thread::spawn(move || {
            // Load the dataset from a the filesystem into an adjacency matrix.
            let adjacency_matrix: Graph = dataset.load(io_bar);

            // run each of the algorithms in their own thread.
            // first the spectral count.
            let spectral_input = adjacency_matrix.clone();
            let spectral_thread =
                thread::spawn(move || AlgorithmBenchmark::run(
                    spectral_count_bar,
                    spectral_count,
                    &spectral_input
                ));

            // Set the gamma for TraceTriangle.
            let gamma = 1f64;
            // Next trace_triangle_r
            let ttr_input = TraceTriangle {
                random_vector_variant: RandomVector::Rademacher,
                seed: None,
                gamma,
                graph: adjacency_matrix.clone()
            };
            let ttr_thread =
                thread::spawn(move || AlgorithmBenchmark::run(
                    ttr_bar,
                    TraceTriangle::run,
                    &ttr_input
                ));

            // then trace_triangle_n
            let ttn_input = TraceTriangle {
                random_vector_variant: RandomVector::Normal,
                seed: None,
                gamma,
                graph: adjacency_matrix.clone(),
            };
            let ttn_thread =
                thread::spawn(move || AlgorithmBenchmark::run(
                    ttn_bar,
                    TraceTriangle::run,
                    &ttn_input
                ));
            
            // Join the spawned threads. Convert the results to serializable
            // records, and collect all of those into one list.
            let mut results: Vec<BenchmarkRecord> = Vec::new();
            let mut ttr: Vec<BenchmarkRecord> = ttr_thread
                .join()
                .expect("TTR failed")
                .to_records(
                    "TraceTriangleR",
                    dataset.path,
                    Some(gamma),
                    None
                );
            results.append(&mut ttr);

            let mut ttn: Vec<BenchmarkRecord> = ttn_thread
                .join()
                .expect("TTN failed")
                .to_records(
                    "TraceTriangleN",
                    dataset.path,
                    Some(gamma),
                    None
                );
            results.append(&mut ttn);

            let mut spectral_count: Vec<BenchmarkRecord> = spectral_thread
                .join()
                .expect("Spectral count failed")
                .to_records(
                    "Spectral Count",
                    dataset.path,
                    None,
                    None,
                );
            results.append(&mut spectral_count);

            // Return the results.
            results
        });

        // Add the join handle to the list of threads.
        threads.push(join_handle);
    }

    // Wait for all status bars on the multi-bar to complete.
    multibar.join().unwrap();

    // Collect all the results from child threads anc save them into a CSV file.
    let mut output: Writer<File> = Writer::from_path("results.csv")
        .expect("Could not make output file.");
    for results_thread in threads {
        let results: Vec<BenchmarkRecord> = results_thread.join()
            .expect("Could not join results thread.");

        // Write each result to the CSV file.
        for result in results {
            output.serialize(result)
                .expect("Could not serialize record.");
        }
    }
}
