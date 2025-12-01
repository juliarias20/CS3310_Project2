import networkx as nx
import random
from Dijkstra import build_example_graph, generate_random_connected_graph, build_weight_dict
import time

def FloydWarshall(graph, weight):

    #Organized nodes to integer list of node_index
    nodes = list(graph.nodes())
    node_index = {nodes[i]: i for i in range(len(nodes))}
    n = len(nodes)

    #Initialize distance matrix of size n x n and set all values to infinity
    distance = [[float('inf')] * len(graph) for _ in range(len(graph))]

    #Fill in edge weights using the node_index
    for (u, v), w in weight.items():
        distance[node_index[u]][node_index[v]] = w

    #Recursion function
    for k in range(n):
        for i in range(n):
            for j in range(n):
                distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])

    return distance, node_index

def time_repeated_floyd(n_values, edge_prob, runs_per_n=5):
    """
    This is the to record the total runtime and divide by the number of runs to compute the average runtime per instance size
    """
    print(f"\n=== Repeated Floyd Warshall timings (edge_prob = {edge_prob}) ===")
    print("{:>5} {:>15}".format("n", "avg_time(s)"))

    for n in n_values:
        total_time = 0.0
        for _ in range(runs_per_n):
            G, weight = generate_random_connected_graph(n, edge_prob)
            start = time.perf_counter()
            _all_dist = FloydWarshall(G, weight)
            end = time.perf_counter()
            total_time += (end - start)
        avg_time = total_time / runs_per_n
        print("{:>5} {:>15.6f}".format(n, avg_time))

if __name__ == "__main__":
    G = build_example_graph()
    weight = build_weight_dict(G)

    # Prints the example grpah to prove function correctness
    print("Floyd-Warshall all-pairs shortest path distances (example graph):")

    distance_matrix, node_index = FloydWarshall(G, weight)

    # Print the distances between each node based on the source
    for u in node_index:
        print(f"Source {u}:")
        currentNode = node_index[u]
        for v in node_index:
            neighbor = node_index[v]
            d = distance_matrix[currentNode][neighbor]
            if d == float('inf'):
                print(f"  to {v}: unreachable")
            else:
                print(f"  to {v}: dist={d}")

    print("ASPS using Floyd-Warshall Algorithm:")
    n_values = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    # Sparse graphs: the 0.2 means that only 20% of possible edges exist
    time_repeated_floyd(n_values, edge_prob=0.2, runs_per_n=5)

    # Dense graphs: the 0.8 means that only 80% of possible edges
    time_repeated_floyd(n_values, edge_prob=0.8, runs_per_n=5)


