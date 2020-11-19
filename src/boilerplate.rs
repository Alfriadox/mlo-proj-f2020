use sprs::{CsMatI};
use csv::ReaderBuilder;
use std::time::Instant;
use crate::Graph;

/// A reference to a csv file in the file system, with information specific
/// to that file that determines how it is parsed.
pub struct Dataset {
    /// The file's path on the filesystem.
    pub path: &'static str,
    /// The ASCII delimiter used while parsing the CSV file.
    pub csv_delimiter: u8,
    /// The number of nodes in the represented graph. This must be accurate
    /// for the algorithms to return accurate results.
    pub nodes: usize,
    /// Does the CSV file have a header row with column that does not
    /// contain data?
    pub has_header_row: bool,
    /// The ASCII character (if any) used to mark a row in the CSV file as a
    /// comment row to ignore.
    pub comment_char: Option<u8>
}

impl Dataset {
    /// Load the dataset from the disc into a sparse adjacency matrix.
    ///
    /// The CSV file is expected to have rows representing the edges in the
    /// graph. Each row should be in the format
    /// ```
    /// a, b
    /// ```
    /// where a and b are integers greater than or equal to zero representing
    /// a node in the graph.
    ///
    /// This function will panic if `self.nodes` is less than the actual number
    /// of nodes (if any row contains a number greater than or equal to `self.nodes`).
    pub fn load(&self) -> Graph {
        // Get the start time.
        let start_time = Instant::now();

        // The adjacency matrix should be a square matrix of size NxN.
        let shape = (self.nodes, self.nodes);
        let mut mat: Graph = CsMatI::zero(shape);

        // construct the CSV reader.
        let mut reader = ReaderBuilder::new()
            .delimiter(self.csv_delimiter)
            .has_headers(self.has_header_row)
            .comment(self.comment_char)
            .from_path(self.path)
            .expect("Could not make CSV reader");

        // for each record in the CSV file, add the appropriate edge in the graph.
        for record in reader.deserialize() {
            // get the next record.
            let (a, b): (usize, usize) = record.expect("Bad record");

            // since the graph is undirected, we set two cells in the adjacency
            // matrix.
            mat.insert(a, b, 1.);
            mat.insert(b, a, 1.);
        }

        // let the user know that we finished reading the file (and note how long it took).
        println!("Loaded {} in {}ms.", self.path, start_time.elapsed().as_millis());

        // return the created matrix.
        return mat;
    }
}
