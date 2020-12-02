use crate::boilerplate::Dataset;
use sprs::CsMat;
use crate::processing::{AlgorithmBenchmark, BenchmarkRecord};
use crate::algs::{spectral_count, TraceTriangle, RandomVector, EigenTriangle};
use std::thread;
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
pub const TRIALS: u8 = 50;

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
    Dataset {
        path: "data/CA-GrQc/CA-GrQc.txt",
        nodes: 5_242,
        csv_delimiter: b'\t',
        has_header_row: false,
        comment_char: Some(b'#')
    },
    Dataset {
        path: "data/CA-AstroPh/CA-AstroPh.txt",
        nodes: 18_772,
        csv_delimiter: b'\t',
        has_header_row: false,
        comment_char: Some(b'#')
    },
];

/// The type used to represent graphs. This is currently an adjacency matrix
/// with boolean cells.
pub type Graph = CsMat<bool>;

/// The type used to represent triangle counts (or estimates). This is currently
/// a 32-bit unsigned integer.
pub type TriangleEstimate = u32;

fn main() {
    // Create a list of thread handles to receive results from on completion.
    let mut threads: Vec<JoinHandle<Vec<BenchmarkRecord>>> = Vec::with_capacity(DATASETS.len());

    for dataset in DATASETS {
        // Load the dataset from a the filesystem into an adjacency matrix.
        let join_handle: JoinHandle<Vec<BenchmarkRecord>> = thread::spawn(move || {
            let mut results: Vec<BenchmarkRecord> = Vec::with_capacity(4*TRIALS as usize);

            let adjacency_matrix: Graph = dataset.load();

            // run each of the algorithms in their own thread.
            // first the spectral count.
            let spectral_input = adjacency_matrix.clone();
            let spectral_thread = thread::spawn(move || {
                AlgorithmBenchmark::run(
                    dataset,
                    "SpectralCount",
                    spectral_count,
                    &spectral_input
                ).to_records(None, None)
            });

            // Set the gamma for TraceTriangle.
            let gamma = 1f64;

            // Next trace_triangle_r
            let ttr_input = TraceTriangle {
                random_vector_variant: RandomVector::Rademacher,
                seed: None,
                gamma,
                graph: adjacency_matrix.clone()
            };

            let ttr_thread = thread::spawn(move || {
                AlgorithmBenchmark::run(
                    dataset,
                    "TraceTriangleR",
                    TraceTriangle::run,
                    &ttr_input
                ).to_records(Some(gamma), None)
            });

            // Then trace_triangle_n.
            let ttn_input = TraceTriangle {
                random_vector_variant: RandomVector::Normal,
                seed: None,
                gamma,
                graph: adjacency_matrix.clone(),
            };

            let ttn_thread = thread::spawn(move || {
                AlgorithmBenchmark::run(
                    dataset,
                    "TraceTriangleN",
                    TraceTriangle::run,
                    &ttn_input
                ).to_records(Some(gamma), None, )
            });

            // Lastly EigenTriangle.
            let max_iters: usize = 20;
            let eigen_input = EigenTriangle {
                maximum_iterations: max_iters,
                graph: adjacency_matrix.clone()
            };

            let eigen_thread = thread::spawn(move || {
                AlgorithmBenchmark::run(
                    dataset,
                    "EigenTriangle",
                    EigenTriangle::run,
                    &eigen_input
                ).to_records(None, Some(max_iters))
            });

            // Return the generated results.
            results.append(&mut spectral_thread.join().expect("Spectral thread failed."));
            results.append(&mut ttr_thread.join().expect("TTR failed"));
            results.append(&mut ttn_thread.join().expect("TTN Failed"));
            results.append(&mut eigen_thread.join().expect("EigenTri Failed"));
            results
        });

        // Add the results thread to the list of executing threads.
        threads.push(join_handle);
    }

    // Collect all the results from child threads anc save them into a CSV file.
    let mut output: Writer<File> = Writer::from_path("results.csv")
        .expect("Could not make output file.");
    for thread_handle in threads {
        let results = thread_handle.join()
            .expect("Could not join results thread.");

        // Write each result to the CSV file.
        for result in results {
             output.serialize(result)
                .expect("Could not serialize record.")
        }
    }
}
