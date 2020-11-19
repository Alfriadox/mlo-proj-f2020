use crate::boilerplate::Dataset;
use sprs::CsMat;
use crate::processing::AlgorithmBenchmark;
use crate::algs::spectral_count;

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
];

/// The type used to represent graphs. This is currently an adjacency matrix
/// with boolean cells.
pub type Graph = CsMat<bool>;

/// The type used to represent triangle counts (or estimates). This is currently
/// a 32-bit unsigned integer.
pub type TriangleEstimate = u32;

fn main() {
    // for each of the datasets.
    DATASETS.into_iter()
        .for_each(move |dataset: &Dataset| {
            // Load the dataset from a the filesystem into an adjacency matrix.
            let adjacency_matrix: Graph = dataset.load();
            let spectral_count = AlgorithmBenchmark::run(
                format!("Spectral Count ({})", dataset.path),
                spectral_count,
                &adjacency_matrix
            );

        })
}
