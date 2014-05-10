import networkx as nx
import random
import pylab
import math
import matplotlib.backends.backend_pdf


# Graph related function
def random_connected_graph(n, e, directed=False, seed=None):
    """Return a random connected graph.

    Parameters
    ----------
    n : int
        The number of nodes.
    e : int
        The number of edges.
    seed : int, optional
        Seed for random number generator (default=None).
    directed : bool, optional (default=False)
        If True return a directed graph.

    Notes
    -----
    The function returns random connected graph. We define a connected graph
    as one where there are no disconnected subgraphs. If the method returns
    directed graph, then the graph is strongly connected.

    The function requires e>n to construct a minimal connected graph.
    The function requires n*(n-1)/2>=e to have no multiple edges between nodes.

    The function is based on generators.random_graphs.random_regular_graph.
    """
    if e < n-1:
        raise nx.NetworkXError("e<n-1, choose larger e or smaller n")
    if n*(n-1)/2 < e:
        raise nx.NetworkXError("n*(n-1)/2<e, choose larger n or smaller e")
    if seed is not None:
        random.seed(seed)
    # We currently construct only undirected graphs.
    # TODO: Something to think about
    G = nx.empty_graph(n)
    while e > 0:
        connected_components = nx.connected_components(G)
        if len(connected_components) > 1:
            subgraph1 = random.choice(connected_components)
            subgraph2 = random.choice(connected_components)
            while subgraph2 == subgraph1:
                subgraph2 = random.choice(connected_components)
        else:
            subgraph1 = G.nodes()
            subgraph2 = G.nodes()
        node1 = random.choice(subgraph1)
        node2 = random.choice(subgraph2)
        while node2 == node1 or G.has_edge(node1, node2):
            node1 = random.choice(subgraph1)
            node2 = random.choice(subgraph2)
        G.add_edge(node1, node2)
        e -= 1
    if directed:
        G = nx.DiGraph(G)
    return G


# Graph related function
def add_random_weights(G, min_weight, max_weight, seed=None):
    """Add random weights to edges.

    Parameters
    ----------
    G : Graph object
        The graph to add random weights.
    min_weight : int
        Minimum weight
    max_weight : int
        Maximum weight
    seed : int, optional
        PRG seed
    """
    if seed is not None:
        random.seed(seed)
    for edge in G.edges_iter(data=True):
        if "weight" in edge[2]:
            continue
        G.edge[edge[0]][edge[1]]["weight"] = random.randint(min_weight,
                                                            max_weight)
    return G


# Graph algorithm
def random_matching(G, seed=None):
    """
    Returns random matching in a graph.
    """
    M = nx.Graph(G)
    if seed is not None:
        random.seed(seed)
    edges = {}
    while len(M.edges()) > 0:
        edge = random.choice(M.edges())
        if G[edge[0]][edge[1]]['weight'] <= G[edge[1]][edge[0]]['weight']:
            edges[edge[1]] = edge[0]
        else:
            edges[edge[0]] = edge[1]
        for removable_edge in M.edges(nbunch=edge):
            M.remove_edge(*removable_edge)
    return edges


# Graph algorithm
def max_weight_matching(G, maxcardinality=False):
    if type(G) == nx.classes.graph.Graph:
        return nx.algorithms.max_weight_matching(G, maxcardinality)
    else:
        matchings = nx.algorithms.max_weight_matching(
            undirect_graph(G), maxcardinality)
        return undirect_matching(G, matchings)


def undirect_matching(G, matchings):
    matchings = dict(filter(lambda x:
                            G[x[0]][x[1]]['weight'] >=
                            G[x[1]][x[0]]['weight'],
                            matchings.items()))
    newmatchings = {}
    for p in matchings.items():
        if p[1] not in newmatchings:
            newmatchings[p[0]] = p[1]
    return newmatchings


# My project (init)
class File:
    def __init__(self, *args):
        if len(args) > 0:
            self.file = args[0]
        if len(args) > 1:
            self.total = int(math.log10(args[1]-1)) + 1
        else:
            self.total = 1

    def __repr__(self):
        return ("x{0:0>%d}" % (self.total)).format(self.file)

    def __str__(self):
        return ("x{0:0>%d}" % (self.total)).format(self.file)


# My project (init)
def return_n_files(n):
    files = []
    for i in range(n):
        files.append(File(i, n))
    return files


# My project (init)
def add_random_files(G, files, min_weight, max_weight, seed=None):
    """Add random files to nodes.

    Parameters
    G : Graph object
        The graph to add random files to.
    files : list
        List of files which are added to nodes.
    min_weight : int
        Minimum symmetric difference between nodes.
    max_weight : int
        Maximum symmetric difference between nodes.
    seed : int, optional
        PRG seed (default=None)
    """
    if seed is not None:
        random.seed(seed)
    for node in G.nodes_iter():
        no_node_files = len(files) - random.randint(min_weight,
                                                    (max_weight+1)/2)
        node_files = set(random.sample(files, no_node_files))
        G.node[node]["files"] = node_files
    calculate_weights_from_difference(G)
    return G


def undirect_graph(G):
    """
    Undirects graph in a necessary way (considering the weights).
    """
    G = nx.Graph(G)
    calculate_weights_from_difference(G)
    return G


# My project (reconciliation)
def calculate_weights_from_difference(G):
    """
    Calculates edge weights from symmetric difference of node's files at
    endpoints.
    """
    for node1, node2 in G.edges():
        if type(G) == nx.DiGraph:
            weight = len(G.node[node1]["files"].difference(
                         G.node[node2]["files"]))
        else:
            weight = len(G.node[node1]["files"].symmetric_difference(
                         G.node[node2]["files"]))
        G.edge[node1][node2]["weight"] = weight
    return G


# My project (reconciliation)
def round_matching(G, matching_alg=max_weight_matching,
                   seed=None):
    """
    Performs maximum weighted matching. Reconciles files between matched nodes.
    """
    try:
        matchings = matching_alg(G, seed=seed)
    except TypeError:
        matchings = matching_alg(G)
    for (node1, node2) in matchings.items():
        node1_files = G.node[node1]["files"]
        node2_files = G.node[node2]["files"]
        nodes_files = node1_files.union(node2_files)
        if type(G) == nx.classes.graph.Graph:
            G.node[node1]["files"] = nodes_files
        G.node[node2]["files"] = nodes_files
    G = calculate_weights_from_difference(G)
    return G, matchings


# My project (init)
def random_graph_with_files(n, e, no_files, min_weight, max_weight,
                            directed=False, seed=None):
    G = random_connected_graph(n, e, directed, seed)
    files = return_n_files(no_files)
    G = add_random_files(G, files, min_weight, max_weight, seed)
    G = unique_files(G)
    return G


# My project (init)
def unique_files(G):
    files = G.node[0]['files']
    for node in G.nodes():
        files = files.intersection(G.node[node]['files'])
    for node in G.nodes():
        G.node[node]['files'] = G.node[node]['files'] - files
    return G


# My project (drawing)
def draw_round(G, i=1, draw_pos=None, matchings=None, pp=None):
    if draw_pos is None:
        draw_pos = nx.shell_layout(G)
    if matchings is not None:
        nx.draw_networkx_edges(G, draw_pos, edgelist=matchings.items(),
                               width=3,
                               color="r")
    if pp is not None:
        pp.savefig()
    pylab.figure(i)
    labels = dict([((u, v,), d['weight'])
                   for u, v, d in undirect_graph(G).edges(data=True)])
    nx.draw_networkx_edge_labels(G, draw_pos, labels, label="TEST")
    nx.draw(G, draw_pos, label="test")
    draw_files(G, draw_pos)


# My project (drawing)
def draw_files(G, draw_pos):
    bottom = {"verticalalignment":"bottom",}
    top = {"verticalalignment":"top",}
    left = {"horizontalalignment":"left"}
    right = {"horizontalalignment":"right"}
    for node, pos in draw_pos.items():
        par={"fontsize":9,
             "horizontalalignment":"center",
             "verticalalignment":"center"}
        files = sorted([str(x) for x in list(G.node[node]["files"])])
        grouped_files = []
        for i in range(len(files))[::3]:
            grouped_files.append(",".join(files[i:i+3]))
        if pos[1] > 0.01:
            par.update(bottom)
            pylab.text(pos[0],
                       pos[1]+0.15,
                       "\n".join(grouped_files),
                       **par)
        elif pos[1] < -0.01:
            par.update(top)
            pylab.text(pos[0],
                       pos[1]-0.15,
                       "\n".join(grouped_files),
                       **par)
        else:
            if pos[0] > 0.9:
                par.update(left)
                pylab.text(pos[0]+0.1,
                           pos[1],
                           "\n".join(grouped_files),
                           **par)
            elif pos[0] < -0.9:
                par.update(right)
                pylab.text(pos[0]-0.1,
                           pos[1],
                           "\n".join(grouped_files),
                           **par)


def draw_legend(n,e,seed=None, graphseed=None):
    if seed is None:
        seed = "NA"
    if graphseed is None:
        graphseed = "NA"
    pylab.text(1.3,1.3, "nodes: {0}\nedges: {1}\ngraphseed: {2}\nseed: {3}".
            format(n,e,graphseed,seed), bbox={"facecolor":"red", "alpha":0.5})
        

# My project (reconciliation)
def reconcile_all(n=10, e=25, no_files=200, min_weight=1, max_weight=5,
                  matching_alg=max_weight_matching,
                  G=None, draw=True, directed=False, seed=None,
                  pp=None, graphseed=None):
    """
    My test program.
    """
    def not_reconciled(G):
        return any(map(lambda x: x[2]["weight"] > 0, G.edges_iter(data=True)))
    if G is None:
        G = random_graph_with_files(n, e, no_files, min_weight, max_weight,
                                    directed, seed)
    draw_pos = nx.shell_layout(G)
    i = 1
    random.seed(seed)
    matchings = {}
    while not_reconciled(G):
        if draw:
            draw_round(G, i, draw_pos, matchings, pp)
            draw_legend(G.number_of_nodes(), G.number_of_edges(), seed=seed,
                        graphseed=graphseed)
        # If we use previous seed directly, then we get same results all the
        # time as seeds to round_matching overlap.
        newseed = random.random()
        G, matchings = round_matching(G, matching_alg, newseed)
        i += 1
    if draw:
        draw_round(G, i, draw_pos, matchings, pp)
        if pp is not None:
            pp.savefig()
    return i-1


def test(nodes_min=5, nodes_max=20, rounds1=1000, rounds2=1000,
         directed=False):
    results = []
    for no_nodes in range(nodes_min, nodes_max):
        for no_edges in range(no_nodes-1, no_nodes*(no_nodes-1)/2):
            for seed_1 in range(rounds1):
                G = random_graph_with_files(no_nodes, no_edges, 100, 1, 10,
                                            directed, seed_1)
                G_max = G.copy()
                rounds_max = reconcile_all(G=G_max, seed=seed_1, draw=False)
                for seed_2 in range(rounds2):
                    G_random = G.copy()
                    rounds_random = reconcile_all(G=G_random, seed=seed_2,
                                                  draw=False,
                                                  matching_alg=random_matching)
                    result = {"nodes": no_nodes,
                              "edges": no_edges,
                              "seed_1": seed_1,
                              "seed_2": seed_2,
                              "files": 1000,
                              "min_weight": 1,
                              "max_weight": 10,
                              "rounds_max": rounds_max,
                              "rounds_random": rounds_random,
                              "directed": directed,
                              }
                    if rounds_max > rounds_random:
                        print "Disproven", result
                    results.append(result)
    return results

def make_pdf(df):
    dd = df[df.rounds_random < df.rounds_max]
    for a in dd.iterrows():
        pp = matplotlib.backends.backend_pdf.PdfPages("graph_%d.pdf" % a[0])
        G = random_graph_with_files(a[1]['nodes'], a[1]['edges'], 100, 1, 10,
                                    seed=a[1]['seed_1'])
        reconcile_all(G=G.copy(),pp=pp, graphseed=a[1]['seed_1'])
        pylab.close('all')
        reconcile_all(G=G.copy(),pp=pp, graphseed=a[1]['seed_1'],
                      seed=a[1]['seed_2'], matching_alg=random_matching)
        pp.close()
        pylab.close('all')

