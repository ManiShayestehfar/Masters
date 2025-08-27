import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx


def _auto_positions(G):
    """
    Choose a symmetric layout automatically.
    Tries Graphviz; else uses KK / circular / spring.
    Uses an undirected copy for nicer symmetry.
    """
    U = nx.Graph(G)  # undirected for layout symmetry

    # 1) Try Graphviz layouts if available
    try:
        # Prefer 'dot' for DAGs, else 'neato'
        from networkx.drawing.nx_pydot import graphviz_layout
        if nx.is_directed_acyclic_graph(G):
            return graphviz_layout(U, prog="dot")  # layered symmetry
        return graphviz_layout(U, prog="neato")     # force-directed symmetry
    except Exception:
        pass

    # 2) Special-case bipartite graphs
    try:
        from networkx.algorithms import bipartite
        if bipartite.is_bipartite(U):
            # find a bipartition
            left = next(iter(bipartite.sets(U)))
            return nx.bipartite_layout(U, left)
    except Exception:
        pass

    # 3) If graph is close to a cycle/clique, circular looks best
    if U.number_of_edges() >= U.number_of_nodes():
        return nx.circular_layout(U)

    # 4) General nice default
    return nx.kamada_kawai_layout(U)

def intersection_graph(blocks, title, use_latex=False, layout="auto", seed=7):
    mpl.rcParams["text.usetex"] = False  # avoid external LaTeX
    edges = list(blocks.keys()) if hasattr(blocks, "keys") else list(blocks)
    G = nx.DiGraph()
    G.add_edges_from(edges)

    if layout == "auto":
        pos = _auto_positions(G)
    elif layout == "dot":
        # force Graphviz dot if available
        from networkx.drawing.nx_pydot import graphviz_layout
        pos = graphviz_layout(nx.Graph(G), prog="dot")
    elif layout == "neato":
        from networkx.drawing.nx_pydot import graphviz_layout
        pos = graphviz_layout(nx.Graph(G), prog="neato")
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:  # spring as last resort
        pos = nx.spring_layout(G, seed=seed)

    plt.figure(figsize=(6, 4), dpi=300)

    nx.draw_networkx_edges(
        G, pos,
        arrows=True, arrowstyle='-|>', arrowsize=15,
        width=1, connectionstyle="arc3,rad=0.1", edge_color="black"
    )
    nx.draw_networkx_nodes(
        G, pos, node_size=600, node_color="#E9EEF7",
        edgecolors="blue", linewidths=1
    )

    def fmt(lbl): return f"${lbl}$" if use_latex else str(lbl)
    nx.draw_networkx_labels(G, pos, labels={n: fmt(n) for n in G.nodes()}, font_size=12)

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()