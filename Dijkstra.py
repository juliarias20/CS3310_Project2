import networkx as nx

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
