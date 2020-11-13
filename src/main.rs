use std::path::Path;
use sprs::{CsMat, CsMatI, CsMatBase};
use petgraph::{Graph, Undirected};
use csv::{
    Result as CsvResult,
    ReaderBuilder
};

/// Load an undirected graph from a path using petgraph.
/// Will panic if the number of nodes is too low.
fn load_graph(
    path: impl AsRef<Path>,
    delimiter: u8,
    nodes: usize,
    has_header_row: bool,
) -> CsMat<usize> {
    let mut mat = CsMatI::zero((nodes, nodes));

    let mut reader = ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(has_header_row)
        .from_path(path.as_ref())
        .expect("Could not make reader for file");

    for record in reader.deserialize() {
        let (a, b): (usize, usize) = record.expect("Bad record");
        // since the graph is undirected, we double set in the matrix.
        mat.insert(a, b, 1);
        mat.insert(b, a, 1);
    }

    println!("Loaded {} to matrix. {} rows, {} non-zero cells.", path.as_ref().display(), mat.rows(), mat.nnz());
    mat
}


fn main() {
    let twitch_graph = load_graph(
        "data/twitch/ES/musae_ES_edges.csv",
        b',',
        4_648,
        59_382,
        true
    );
}
