use std::path::PathBuf;
use std::io::Result as IOResult;
use sprs::CsMat;
use petgraph::Graph;

type Scalar = usize;

fn load_graph(path: PathBuf) -> IOResult<CsMat<Scalar>> {
    let graph = Graph::new_undirected();
}


fn main() {

}
