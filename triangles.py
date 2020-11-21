# OLD  FILE, not use.

# IMPORTS
import numpy as np
import networkx as nx


def load_graph(path, data=True, delim=None): 
    """
    Given the path to a csv file containing a row for every edge, 
    parse the data into an adjacency matrix. Each row should have two 
    elements, one for each node in the edge. 

    Parameters 
    ------------------------------------------------------------------ 
    path : string 
        A path to the csv file for the graph. 
    data : list of pairs
        Tuples specifying dictionary key names and types for edge 
        data. 
    delim : string 
        Delimiter string for graph read. 

    Returns 
    ------------------------------------------------------------------ 
    out : csr_matrix 
        The graph's adjacency matrix. 
    """
    with open(path, 'rb') as f: 
        G = nx.read_edgelist(f, data=data, delimiter=delim) 
    # The adjacency list is returned as a csr_matrix as a computational 
    # time improvement since most real graphs will be extremely sparse. 
    # This turns |V|^2 operations into |E| operations which is a huge 
    # improvement. 
    A = nx.to_scipy_sparse_matrix(G) 
    return A 


def gt_count(A): 
    """
    Uses spectral counting to calculate the exact total number of 
    triangles in a graph from its adjaceny matrix. 

    Parameters
    ------------------------------------------------------------------ 
    A : csr_matrix 
        Adjacency matrix of the graph. 

    Returns 
    ------------------------------------------------------------------ 
    out : int 
        Exact total count of triangles in the graph. 
    """
    #cubed = np.linalg.matrix_power( A, 3 )
    cubed = A ** 3 
    trace = cubed.diagonal().sum() 
    return trace // 6  # This will be an integer regardless 


def triangletrace(A, gamma, vec='R', seed=None, interval=None): 
    """
    Uses the TraceTraingle algorithm to approximate the total 
    triangle count of an undirected graph. This is the first 
    and second variations, which use either standard normal 
    random vectors or Rademacher random vectors respectively. 

    Parameters 
    ------------------------------------------------------------------ 
    A : csr_matrix 
        Adjacency matrix of the graph. 
    gamma : float 
        Iteration parameter. 
    vec : string 
        'R' if we're using Rademacher random vectors. 
        'N' if we're using standard normal random vectors. 
    seed : int 
        Seed for numpy random generation. 
    interval : int
        The percentage points at which progress reports are printed. 
    """
    if seed: 
        # Seeding is used to make the normal and Rademacher results 
        # more easily comparable. 
        np.random.seed(seed) 

    n = A.shape[0] # n is the number of vertices. 
    M = np.ceil( gamma * np.power( np.log(n), 2 ) ).astype('int') 
    T = [] 
    rad = vec.lower() in ['r','rad','rademacher'] 

    for i in range(M): 
        # Declaration is necessary to support the two different 
        # initializations. 
        x = None 

        if rad: # Rademacher random vectors. 
            # This is a Bernouli random vector following Bern(0.5) 
            # which is transformed to contain outcomes from {-1,+1}. 
            x = -1 + 2 * np.random.binomial( 1, 0.5, size=(n,) ) 
        else: # Standard normal random vectors.
            x = np.random.normal( size=(n,) ) 

        y = A * x # equivalent to (y = A x). 
        Ti = np.dot(y, A * y) / 6 # equivalent to (T_i = y^T A y / 6).

        T.append(Ti) 

        if interval and i % (M // (100/interval)) == 0: 
            # Prints algorithm 
            print('x = \n{}\n'.format(x))
            print('Algo: {}\nProgress: {}%\n'.format(vec,(100*i)//M))

    # Returns the average of all estimations. 
    # We use np.ceil because these estimates are upperbounded by the 
    # exact count and therefore we should not risk invalidating this 
    # bound. 
    return np.floor( np.mean(T) ).astype('int') 


def main(): 
    #""" 
    path = "datasets/deezer_clean_data/HR_edges.csv" 
    # CSV has headers and comma delimiters (instead of spaces)
    A = load_graph( path, data=[('node1',int),('node2',int)], delim=',' ) 
    #"""
    """
    path = "datasets/facebook/0.edges" 
    A = load_graph( path )
    """
    """
    path = "datasets/test.txt" 
    A = load_graph( path )
    """

    seed = np.random.randint(2**32 - 1)

    print( 'SHAPE: {}\nTRIANGLES: {}\nTTN: {}\nTTR: {}'.format( 
        A.shape, 
        gt_count(A), 
        triangletrace(A, 20, vec='N', seed=seed, interval=20), 
        triangletrace(A, 20, vec='R', seed=seed, interval=20) ) ) 

if __name__ == '__main__': 
    main()