use std::path::Path;
use sprs::{CsMat, CsMatI};
use csv::{
    ReaderBuilder
};

/// Load an undirected graph from a path using petgraph.
/// Will panic if the number of nodes is too low.
fn load_graph(
    path: impl AsRef<Path>,
    delimiter: u8,
    nodes: usize,
    has_header_row: bool,
    comment_char: Option<u8>
) -> CsMat<usize> {
    let mut mat = CsMatI::zero((nodes, nodes));

    let mut reader = ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(has_header_row)
        .comment(comment_char)
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

/// Use spectral counting to get the exact number of triangles in the graph.
fn spectral_count(graph: CsMat<usize>) -> usize {
    // Cube the graph's adjacency matrix.
    let cubed: CsMat<usize> = &(&graph * &graph) * &graph;
    // Take the diagonal sum using an iterator.
    let diag_sum = cubed.iter()
        .filter(|(_, (r, c))| r == c)
        .fold(0, |acc, (item, _)| acc+item);
    diag_sum/6
}

struct TestCase {
    path: &'static str,
    csv_delimiter: u8,
    nodes: usize,
    has_header_row: bool,
    comment_char: Option<u8>
}

impl TestCase {
    fn run(self) {
        let graph = load_graph(
            self.path,
            self.csv_delimiter,
            self.nodes,
            self.has_header_row,
            self.comment_char
        );
        
    }
}

fn main() {
    let start_time = std::time::Instant::now();
    let twitch_graph = load_graph(
        "data/twitch/ES/musae_ES_edges.csv",
        b',',
        4_648,
        true,
        None
    );
    let twitch_count = spectral_count(twitch_graph);
    println!(
        "Spectral count of triangles in twitch graph: {}. Time: {} millis",
        twitch_count,
        start_time.elapsed().as_millis()
    );
}
