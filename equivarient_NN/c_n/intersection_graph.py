import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sympy as sp
from scipy.special import erf
import math
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
        from networkx.drawing.nx_pydot import graphviz_layout
        if nx.is_directed_acyclic_graph(G):
            return graphviz_layout(U, prog="dot")   # layered symmetry
        return graphviz_layout(U, prog="neato")     # force-directed symmetry
    except Exception:
        pass

    # 2) Special-case bipartite graphs
    try:
        from networkx.algorithms import bipartite
        if bipartite.is_bipartite(U):
            left = next(iter(bipartite.sets(U)))
            return nx.bipartite_layout(U, left)
    except Exception:
        pass

    # 3) If graph is close to a cycle/clique, circular looks best
    if U.number_of_edges() >= U.number_of_nodes():
        return nx.circular_layout(U)

    # 4) General nice default
    return nx.kamada_kawai_layout(U)

def intersection_graph(
    blocks,
    title,
    use_latex=False,
    layout="auto",
    seed=7,
    ax=None,
    figsize=(4, 4),
    dpi=300,
    show=False,
    return_pos=False,
    node_style=None,
    edge_style=None,
):
    mpl.rcParams["text.usetex"] = bool(use_latex)
    mpl.rcParams["mathtext.fontset"] = "cm"

    edges = list(blocks.keys())
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # Choose layout
    if layout == "auto":
        pos = _auto_positions(G)
    elif layout in {"dot", "neato"}:
        try:
            from networkx.drawing.nx_pydot import graphviz_layout
            prog = "dot" if layout == "dot" else "neato"
            pos = graphviz_layout(nx.Graph(G), prog=prog)
        except Exception:
            pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed)

    # Figure / axis
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
        created_fig = True
    else:
        fig = ax.figure

    # Styles
    one_way_kw = dict(
        arrows=True, arrowstyle="-|>", arrowsize=15,
        width=1, connectionstyle="arc3,rad=0.1", edge_color="black"
    )
    two_way_kw = dict(
        arrows=True, arrowstyle="<|-|>", arrowsize=15,
        width=1.3, connectionstyle="arc3,rad=0.0", edge_color="darkblue"
    )

    if edge_style: one_way_kw.update(edge_style)
    node_kw = dict(node_size=600, node_color="#E9EEF7",
                   edgecolors="blue", linewidths=1)
    if node_style: node_kw.update(node_style)

    # Partition edges into one-way vs two-way
    drawn_two_way = set()
    one_way_edges = []
    two_way_edges = []

    for u, v in G.edges():
        if u == v:
            # self-loop: always one-way
            one_way_edges.append((u, v))
        elif (v, u) in G.edges():
            if (v, u) not in drawn_two_way and (u, v) not in drawn_two_way:
                two_way_edges.append((u, v))
                drawn_two_way.add((u, v))
                drawn_two_way.add((v, u))
        else:
            one_way_edges.append((u, v))

    # Draw edges
    if one_way_edges:
        nx.draw_networkx_edges(G, pos, edgelist=one_way_edges, ax=ax, **one_way_kw)
    if two_way_edges:
        nx.draw_networkx_edges(G, pos, edgelist=two_way_edges, ax=ax, **two_way_kw)

    # Draw nodes + labels
    nx.draw_networkx_nodes(G, pos, ax=ax, **node_kw)
    def fmt(lbl): return f"${lbl}$" if use_latex else str(lbl)
    nx.draw_networkx_labels(G, pos, labels={n: fmt(n) for n in G.nodes()}, font_size=12, ax=ax)

    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    fig.tight_layout()

    if show:
        plt.show()

    if return_pos:
        return fig, ax, pos
    return fig, ax
