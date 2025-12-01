import networkx as nx
from Dijkstra import build_example_graph, generate_random_connected_graph, build_weight_dict
import time

def FloydWarshall(graph, weight):
    distance = {}
    previous = {}

    # List of nodes
    nodes = list(graph.nodes())

    # Initialize distance dictionary with infinities
    distance = {u: {v: float('inf') for v in nodes} for u in nodes}

    #For each node distance to itself, set distance to 0
    for u in nodes:
        distance[u][u] = 0

    for u in graph: # Initialize distance matrix with given weights
        for v in graph:
            distance[u][v] = weight.get((u,v),float('inf'))

    #Recursion function
    for k in nodes:
        for i in nodes:
            for j in nodes:
                distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])

    return distance

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

    distance_matrix = FloydWarshall(G, weight)

    # Print the distances between each node based on the source
    for u in distance_matrix:
        print(f"Source {u}:")
        for v in distance_matrix:
            #Analyze each neighbor and use the floyd-warshall algorithm to find distance at this point
            d = distance_matrix[u][v]
            if d == float('inf'):
                print(f"  to {v}: unreachable")
            else:
                print(f"  to {v}: dist = {d}")

    print("ASPS using Floyd-Warshall Algorithm:")
    n_values = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    # Sparse graphs: the 0.2 means that only 20% of possible edges exist
    time_repeated_floyd(n_values, edge_prob=0.2, runs_per_n=5)

    # Dense graphs: the 0.8 means that only 80% of possible edges
    time_repeated_floyd(n_values, edge_prob=0.8, runs_per_n=5)


