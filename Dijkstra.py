import networkx as nx
import random
import time

# Dijkstra's algorithm for single-source shortest paths
# Translated from given pseudocode in Greedy slides
def DijkstraSP(graph,weight, start):
    dist = {}
    prev = {}
    mark = {}
    for v in graph:
        dist[v] = weight.get((start,v),float('inf'))
        prev[v] = start
        mark[v] = 0

    dist[start] = 0
    prev[start] = None
    mark[start] = 1
    # loop n - 1 times
    for i in range(len(graph) - 1):
        # Translated: 𝑢 is the vertex s.t. 𝑚𝑎𝑟𝑘 𝑢 = 0 and 𝑑𝑖𝑠𝑡[𝑢] is minimum
        u = min((v for v in dist if mark[v] == 0), key=lambda x: dist[x]) # Translated outcome given by AI
        mark[u] = 1
        # for each neight v of u s.t. mark[v] = 0
        for v in graph[u]:
            if mark[v] == 0:
                # Relax edge (u,v)
                if dist[v] > dist[u] + weight.get((u,v),float('inf')):
                    dist[v] = dist[u] + weight.get((u,v),float('inf'))
                    prev[v] = u
    return dist, prev


def build_example_graph():
    """
    This is the example that proves the correctness of our implementation. 
    """

    G = nx.DiGraph()
    # this is the source node, target node and the weight
    edges = [
        ('A', 'B', 4),
        ('A', 'C', 2),
        ('B', 'C', 5),
        ('B', 'D', 10),
        ('C', 'E', 3),
        ('E', 'D', 4),
        ('D', 'F', 11),
    ]
    # this will loop through the edges and add the graph object G
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    return G

def build_weight_dict(graph):
    """
    this creates a table for the edge weights instead of seearching for the 
    edge weight in the graph object.
    """
    weight = {}
    for u, v, data in graph.edges(data=True):
        w = data.get('weight', 1.0)
        weight[(u, v)] = float(w)
    return weight

def repeated_dijkstra_all_pairs(graph, weight):
    """
    This helps repeat the dijkstra algorithm for every vertex. 
    """
    all_dist = {}
    all_prev = {}
    for s in graph:  # each vertex as source
        dist, prev = DijkstraSP(graph, weight, s)
        all_dist[s] = dist
        all_prev[s] = prev
    return all_dist, all_prev

def reconstruct_path(prev, s, t):
    """
   This is the extra credit part of the assignment. This will return the shortest path between
   any pair of vertices. 
    """
    path = []   #the path nodes
    cur = t     #start of the target

    while cur is not None:  
        path.append(cur)
        if cur == s:
            break
        cur = prev[cur]

    # if not path: when t is None
    # path[-1] !- s: checks for disconnected graphs
    if not path or path[-1] != s:   
        return []  # unreachable
    path.reverse()
    return path

def generate_random_connected_graph(n, edge_prob, low=1, high=10):
    """
    This will test the algorithm with different dense and sparse with different number of vertices

    """
    G = nx.DiGraph()
    G.add_nodes_from(range(n)) # creates the number of nodes from 0 to n-1
    weight = {}

    # this helps make a path from any node to reach any other node 
    for u in range(n):
        v = (u + 1) % n
        w = random.randint(low, high)
        G.add_edge(u, v, weight=w)
        weight[(u, v)] = float(w)
    # This the density part
    #it will iterate through every pair of nodes, in order to see a shortcut
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            if (u, v) in weight:
                continue
            if random.random() < edge_prob:
                w = random.randint(low, high)
                G.add_edge(u, v, weight=w)
                weight[(u, v)] = float(w)

    return G, weight

def time_repeated_dijkstra(n_values, edge_prob, runs_per_n=5):
    """
    This is the to record the total runtime and divide by the number of runs to compute the average runtime per instance size
    """
    print(f"\n=== Repeated Dijkstra APSP timings (edge_prob = {edge_prob}) ===")
    print("{:>5} {:>15}".format("n", "avg_time(s)"))

    for n in n_values:
        total_time = 0.0
        for _ in range(runs_per_n):
            G, weight = generate_random_connected_graph(n, edge_prob)
            start = time.perf_counter()
            _all_dist, _all_prev = repeated_dijkstra_all_pairs(G, weight)
            end = time.perf_counter()
            total_time += (end - start)
        avg_time = total_time / runs_per_n
        print("{:>5} {:>15.6f}".format(n, avg_time))

if __name__ == "__main__":
    # this is to test the example graph
    G = build_example_graph()
    weight = build_weight_dict(G)
    all_dist, all_prev = repeated_dijkstra_all_pairs(G, weight)

    print("All-pairs shortest path distances (example graph):")
    for s in G.nodes():
        print(f"Source {s}:")
        for t in G.nodes():
            d = all_dist[s][t]
            if d == float('inf'):
                print(f"  to {t}: unreachable")
            else:
                #the extra credit part of the assignment
                path = reconstruct_path(all_prev[s], s, t)
                print(f"  to {t}: dist = {d}, path = {path}")
        print()

    # This is the runtime part of the assignment
    n_values = [10, 20, 30, 40, 50, 100]

    # Sparse graphs: the 0.2 means that only 20% of possible edges exist
    time_repeated_dijkstra(n_values, edge_prob=0.2, runs_per_n=5)

    # Dense graphs: the 0.8 means that only 80% of possible edges
    time_repeated_dijkstra(n_values, edge_prob=0.8, runs_per_n=5)
