import networkx as nx
import random
from Dijkstra import build_example_graph, build_weight_dict

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
            distance[u][v] = weight[u][v]

    # Recursion Function 
    for k in range(n):
        for u in range(n):
            for v in range(n):
                distance[u][v] = min(distance[u][v], distance[u][k] + distance[k][v])

    return distance


# if __name__ == "__main__":
#     # this is to test the example graph
#     G = build_example_graph()
#     weight - build_weight_dict(6)
#     graph = nx.DiGraph()

#     # Generate random matrix of weighted edges (node1, node2, weight)


#     # Add edges using function add_weighted_edges_from
#     graph.add_weighted_edges_from() #weight matrix
    
#     FloydWarshall(graph, x) #weight matrix

