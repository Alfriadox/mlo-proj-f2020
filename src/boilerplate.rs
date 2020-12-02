use sprs::{CsMatI};
use csv::ReaderBuilder;
use crate::Graph;
use indicatif::ProgressBar;
use std::fs::File;

/// A reference to a csv file in the file system, with information specific
/// to that file that determines how it is parsed.
#[derive(Debug)]
pub struct Dataset {
    /// The file's path on the filesystem.
    pub path: &'static str,
    /// The ASCII delimiter used while parsing the CSV file.
    pub csv_delimiter: u8,
    /// The number of nodes in the represented graph. This must be accurate
    /// for the algorithms to return accurate results.
    pub nodes: usize,
    /// The number of edges in the represented graph. This is added directly to
    /// the CSV file and is not used in calculations.
    pub edges: usize,
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
    pub fn load(&self, progress_bar: ProgressBar) -> Graph {
        // The adjacency matrix should be a square matrix of size NxN.
        let shape = (self.nodes, self.nodes);
        // we use bytes here until we do the transpose at the end.
        let mut mat: Graph = CsMatI::zero(shape);

        let file_reader = File::open(self.path)
            .expect("Could not open file");

        // construct the CSV reader.
        let mut reader = ReaderBuilder::new()
            .delimiter(self.csv_delimiter)
            .has_headers(self.has_header_row)
            .comment(self.comment_char)
            // wrap the progress bar here
            .from_reader(progress_bar.wrap_read(file_reader));

        // for each record in the CSV file, add the appropriate edge in the graph.
        for record in reader.deserialize() {
            // get the next record.
            let (a, b): (usize, usize) = record.expect("Bad record");

            // Set the edge in the adjacency matrix.
            mat.insert(a, b, true);
            mat.insert(b, a, true);
        }

        // Finish the progress bar when we are done reading.
        progress_bar.finish_and_clear();

        // Return the created matrix.
        return mat;
    }
}
