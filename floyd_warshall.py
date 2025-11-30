import networkx as nx
import random
from Dijkstra import build_example_graph, generate_random_connected_graph
import time

def FloydWarshall(graph, weight):

    """
    Pseudocode (from lecture slides):
        for u = 1 to n do:

            Array D0, for every u,v = weight(u,v)
        for k = 1 to n do:
            for u = 1 to n do:
                for v = 1 to n do:
                    Array Dk = min(Dk-1, Dk-1 from u to k + Dk-1 from k to v)
        
        return Dk 
    """

    distance = [[float('inf')] * len(graph) for _ in range(len(graph))] # Initialize distance matrix of size n x n and set all values to infinity
    n = len(graph)

    for u in range(n): # Initialize distance matrix with given weights
        for v in range(n):
            distance[u][v] = weight.get((u,v),float('inf'))

    # Recursion Function 
    for k in range(n):
        for u in range(n):
            for v in range(n):
                distance[u][v] = min(distance[u][v], distance[u][k] + distance[k][v])

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
    print("ASPS using Floyd-Warshall Algorithm:")
    n_values = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    # Sparse graphs: the 0.2 means that only 20% of possible edges exist
    time_repeated_floyd(n_values, edge_prob=0.2, runs_per_n=5)

    # Dense graphs: the 0.8 means that only 80% of possible edges
    time_repeated_floyd(n_values, edge_prob=0.8, runs_per_n=5)


