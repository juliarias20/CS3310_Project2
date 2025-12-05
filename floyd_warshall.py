import networkx as nx
from Dijkstra import build_example_graph, generate_random_connected_graph, build_weight_dict
import time


def FloydWarshall(graph, weight):
    """
    Floyd-Warshall algorithm for all pairs shortest paths --- translated from pseudocode in Dynamic Programming slides
    
    This algorithm calculates the shortest path from all pairs, 
    returning a final shortest distance matrix and a matrix of predecessors
    """
    # List of nodes to track the amount
    nodes = list(graph.nodes())

    # Initialize distance dictionary with infinities
    distance = {}
    for u in nodes:
        distance[u] = {}
        for v in nodes:
            distance[u][v] = float('inf')

    # For each node, distance to itself is 0
    for u in nodes:
        distance[u][u] = 0

    # Initialize predecessor matrix
    # previous[u][v] = immediate predecessor of v on shortest path from u to v
    previous = {}
    for u in nodes:
        previous[u] = {}
        for v in nodes:
            previous[u][v] = None

    # Initialize distance matrix with given edge weights
    # Also initialize predecessor ---> if edge (u,v) exists, predecessor of v is u
    for u in nodes:
        for v in nodes:
            if u != v:
                # Check if edge exists, otherwise use infinity
                if (u, v) in weight:
                    edge_weight = weight[(u, v)]
                else:
                    edge_weight = float('inf')

                distance[u][v] = edge_weight

                # If an edge exists, set the predecessor
                if edge_weight != float('inf'):
                    previous[u][v] = u

    # Main Floyd-Warshall relaxation (the core algorithm)
    for k in nodes:
        for i in nodes:
            for j in nodes:
                # Check if path through k (intermediate node) is shorter
                new_distance = distance[i][k] + distance[k][j]
                if new_distance < distance[i][j]: # If the distance using k is shorter
                    # Update distance to the shorter path
                    distance[i][j] = new_distance
                    # Predecessor of j on path i->j is same as predecessor of j on path k->j
                    previous[i][j] = previous[k][j]

    return distance, previous


def reconstruct_path_fw(previous, s, t):
    """
    This algorithm reconstructs the path using the predecessor matrix and select source and target nodes.

    From t to s, if a predecessor exists, add to the path. 

    """
    if s == t:
        return [s]

    if previous[s][t] is None:
        return []  # Unreachable

    # Build path backwards from t to s
    path = []
    current = t
    while current is not None and current != s:
        path.append(current)
        current = previous[s][current]

    if current is None:
        return [] # Unreachable code, just in case

    path.append(s)
    path.reverse()
    return path

def time_repeated_floyd(n_values, edge_prob, runs_per_n=5):
    """
    This is the to record the total runtime and divide by the number of runs to compute the average runtime per instance size
    """
    print(f"\n=== Repeated Floyd Warshall timings (edge_prob = {edge_prob}) ===")
    print("{:>5} {:>15}".format("n", "avg_time(s)"))

    for n in n_values:
        total_time = 0.0
        for _ in range(runs_per_n):
            G, weight = generate_random_connected_graph(n, edge_prob) # Create random generated grpah based on size 'n' and edge probability
            
            start = time.perf_counter() # Start runtime timer
            _all_dist, _all_prev = FloydWarshall(G, weight) # Obtain distance and predecessor matrix
            end = time.perf_counter() # End runtime timer
            
            total_time += (end - start) # Calculate total time and average time taken to obtain result
        avg_time = total_time / runs_per_n
        print("{:>5} {:>15.6f}".format(n, avg_time)) 

if __name__ == "__main__":
    """
    Main function --- performs example test and analyzes runtime for sparse/dense graphs of different sizes.
    
    """
    G = build_example_graph()
    weight = build_weight_dict(G)

    # Prints the example graph to prove function correctness
    print("Floyd-Warshall all-pairs shortest path distances (example graph):")

    distance_matrix, previous_matrix = FloydWarshall(G, weight)

    # Print the distances between each node based on the source
    for u in distance_matrix:
        print(f"Source {u}:")
        for v in distance_matrix:
            # Analyze each neighbor and use the floyd-warshall algorithm to find distance at this point
            d = distance_matrix[u][v]
            if d == float('inf'):
                print(f"  to {v}: unreachable")
            else:
                # Extra credit: show the reconstructed path
                path = reconstruct_path_fw(previous_matrix, u, v)
                print(f"  to {v}: dist = {d}, path = {path}")
        print()

    print("APSP using Floyd-Warshall Algorithm:")
    n_values = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    # Sparse graphs: the 0.2 means that only 20% of possible edges exist
    time_repeated_floyd(n_values, edge_prob=0.2, runs_per_n=5)

    # Dense graphs: the 0.8 means that only 80% of possible edges
    time_repeated_floyd(n_values, edge_prob=0.8, runs_per_n=5)


