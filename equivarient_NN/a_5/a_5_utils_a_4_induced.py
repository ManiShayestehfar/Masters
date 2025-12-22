import itertools as it
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# ============================================================
# 0. A5 as even permutations on {0,1,2,3,4}
# ============================================================

def compose(p, q):
    """(p ∘ q)(i) = p(q(i)) for permutations as tuples."""
    return tuple(p[iq] for iq in q)

def inv(p):
    """Inverse permutation."""
    n = len(p)
    out = [0]*n
    for i, pi in enumerate(p):
        out[pi] = i
    return tuple(out)

def sign_of_perm(p):
    """Parity via inversion count."""
    invs = 0
    for i in range(len(p)):
        for j in range(i+1, len(p)):
            invs += (p[i] > p[j])
    return -1 if (invs % 2) else 1

def A5_elements():
    """All 60 even permutations of S5."""
    els = []
    for p in it.permutations(range(5)):
        if sign_of_perm(p) == 1:
            els.append(tuple(p))
    return els


# ============================================================
# 1. Conjugacy classes in A5 (computed, then labelled)
# ============================================================

@dataclass(frozen=True)
class A5Class:
    label: str
    rep: tuple
    members: list

def conjugacy_classes():
    """
    Compute A5 conjugacy classes by brute-force conjugation in A5.
    Label them as 1A, 2A, 3A, 5A, 5B using standard reps.
    """
    G = A5_elements()
    unseen = set(G)
    classes = []

    while unseen:
        g = unseen.pop()
        orb = set()
        for h in G:
            c = compose(compose(h, g), inv(h))
            orb.add(c)
        unseen -= orb
        classes.append(list(orb))

    reps = [min(C) for C in classes]
    rep_to_members = {rep: sorted(C) for rep, C in zip(reps, classes)}

    e = tuple(range(5))
    rep_3 = (1,2,0,3,4)          # (0 1 2)
    rep_2 = (1,0,3,2,4)         # (0 1)(2 3)
    rep_5A = (1,2,3,4,0)         # (0 1 2 3 4)
    rep_5B = inv(rep_5A)         # (0 4 3 2 1)

    def find_class_containing(x):
        for rep, members in rep_to_members.items():
            if x in members:
                return rep
        raise RuntimeError("Class not found")

    rep_1A = find_class_containing(e)
    rep_3A = find_class_containing(rep_3)
    rep_2A = find_class_containing(rep_2)
    rep_5a = find_class_containing(rep_5A)
    rep_5b = find_class_containing(rep_5B)

    labelled = [
        A5Class("1A", rep_1A, rep_to_members[rep_1A]),
        A5Class("3A", rep_3A, rep_to_members[rep_3A]),
        A5Class("2A", rep_2A, rep_to_members[rep_2A]),
        A5Class("5A", rep_5a, rep_to_members[rep_5a]),
        A5Class("5B", rep_5b, rep_to_members[rep_5b]),
    ]

    sizes = sorted([len(C.members) for C in labelled])
    if sizes != sorted([1, 20, 15, 12, 12]):
        raise RuntimeError(f"Unexpected class sizes: {sizes}")

    return labelled


# ============================================================
# 2. Induced permutation representation: A5 acts on cosets A5/A4
#    Take A4 = Stab_{A5}(4) = { g in A5 : g(4)=4 } (order 12, index 5)
# ============================================================

def A4_subgroup_stab_of_4():
    """A4 as the stabiliser of point 4 inside A5."""
    G = A5_elements()
    H = [g for g in G if g[4] == 4]
    if len(H) != 12:
        raise RuntimeError(f"Expected |A4|=12, got {len(H)}. Something is off.")
    return H

def left_cosets(G, H):
    """
    Compute left cosets of H in G.
    Returns:
      reps   : list of chosen representatives r_i
      cosets : list of cosets as frozensets
      where coset_i = r_i H = { r_i ∘ h : h in H }.
    """
    Hset = set(H)
    unseen = set(G)
    reps = []
    cosets = []

    while unseen:
        r = unseen.pop()
        C = frozenset(compose(r, h) for h in Hset)
        reps.append(r)
        cosets.append(C)
        unseen -= set(C)

    return reps, cosets

def coset_index_map(cosets):
    """Map each coset (frozenset) to an index."""
    return {C: i for i, C in enumerate(cosets)}

def action_on_left_cosets(G, H, cosets, g):
    """
    Given g in G, compute the permutation of cosets induced by left multiplication:
      g · (xH) = (g x)H.
    Returns a list perm of length m where perm[j] = i meaning g sends coset j to coset i.
    """
    idx = coset_index_map(cosets)
    m = len(cosets)
    perm = [0]*m
    for j, C in enumerate(cosets):
        # pick any representative x in the coset C
        x = next(iter(C))
        gx = compose(g, x)
        # compute (gx)H
        newC = frozenset(compose(gx, h) for h in H)
        perm[j] = idx[newC]
    return perm

def perm_matrix_on_cosets(G, H, cosets, g):
    """
    Permutation matrix of g acting on the basis {cosets}.
    Column j is sent to row perm[j].
    """
    perm = action_on_left_cosets(G, H, cosets, g)
    m = len(cosets)
    M = np.zeros((m, m), dtype=float)
    for j, i in enumerate(perm):
        M[i, j] = 1.0
    return M


# ============================================================
# 3. Character table of A5 (standard labelling)
# ============================================================

def character_table():
    sqrt5 = np.sqrt(5.0)
    phi  = (1.0 + sqrt5)/2.0
    phi2 = (1.0 - sqrt5)/2.0
    return {
        "1":  {"1A": 1, "3A": 1,  "2A": 1,  "5A": 1,   "5B": 1},
        "3":  {"1A": 3, "3A": 0,  "2A": -1, "5A": phi, "5B": phi2},
        "3p": {"1A": 3, "3A": 0,  "2A": -1, "5A": phi2,"5B": phi},
        "4":  {"1A": 4, "3A": 1,  "2A": 0,  "5A": -1,  "5B": -1},
        "5":  {"1A": 5, "3A": -1, "2A": 1,  "5A": 0,   "5B": 0},
    }

def irrep_dimension(irrep_label):
    return {"1": 1, "3": 3, "3p": 3, "4": 4, "5": 5}[irrep_label]


# ============================================================
# 4. Projectors via character-sum formula (A5-specific)
#     BUT NOW for the induced coset representation A5 acting on A5/A4
# ============================================================

def projectors_on_coset_rep(tol=1e-10):
    """
    Build projectors p_irrep on V = R[A5/A4] (dimension 5),
    where A4 = Stab_{A5}(4), using:
      p_χ = (d_χ/|A5|) Σ_{g∈A5} χ(g^{-1}) ρ(g).

    Returns:
      proj   : dict irrep_label -> (m×m) numpy matrix
      Qblocks: dict irrep_label -> (m×r) basis columns for image(proj)
      cosets : list of cosets (the basis objects)
      classes: conjugacy classes (labelled)
    """
    G = A5_elements()
    H = A4_subgroup_stab_of_4()
    _, cosets = left_cosets(G, H)
    m = len(cosets)
    if m != 5:
        raise RuntimeError(f"Expected index 5, got {m}")

    classes = conjugacy_classes()
    chartab = character_table()
    group_order = 60

    # Precompute rho(g) on cosets
    rho = {g: perm_matrix_on_cosets(G, H, cosets, g) for g in G}

    # Precompute class sums Σ_{g∈C} rho(g)
    class_sums = {}
    for C in classes:
        S = np.zeros((m, m), dtype=float)
        for g in C.members:
            S += rho[g]
        class_sums[C.label] = S

    proj = {}
    Qblocks = {}

    for ir, chi_ir in chartab.items():
        d = irrep_dimension(ir)
        M = np.zeros((m, m), dtype=float)

        for C in classes:
            chi = float(chi_ir[C.label])  # χ constant on the class
            M += chi * class_sums[C.label]

        p = (d / group_order) * M
        p = 0.5 * (p + p.T)  # symmetrise for numerical stability
        proj[ir] = p

        vals, vecs = np.linalg.eigh(p)
        idx1 = np.where(np.abs(vals - 1.0) < tol)[0]
        Qblocks[ir] = vecs[:, idx1]

    return proj, Qblocks, cosets, classes


# ============================================================
# 5. Interaction graph on this induced representation
# ============================================================

def test_edge_activation(Qblocks, ir_a, ir_b, activation_fn, tol=1e-8):
    Qa = Qblocks.get(ir_a)
    Qb = Qblocks.get(ir_b)
    if Qa is None or Qb is None or Qa.size == 0 or Qb.size == 0:
        return False

    phi = np.vectorize(activation_fn)

    def leaks(v):
        v_act = phi(v)
        coeffs = Qb.T @ v_act
        return np.any(np.abs(coeffs) > tol)

    for a in range(Qa.shape[1]):
        if leaks(Qa[:, a]):
            return True

    for a in range(Qa.shape[1]):
        for b in range(a+1, Qa.shape[1]):
            if leaks(Qa[:, a] + Qa[:, b]):
                return True

    return False


def build_interaction_graph(activation_fn, tol=1e-8, verbose=False):
    """
    Interaction graph inside V = Ind_{A4}^{A5}(1) ≅ R[A5/A4] (dim 5).
    Nodes are A5 irreps: 1, 3, 3', 4, 5 that actually appear in this rep.
    """
    _, Qblocks, _, _ = projectors_on_coset_rep()

    irreps_present = [ir for ir, Q in Qblocks.items() if Q.shape[1] > 0]
    if verbose:
        dims = {ir: Qblocks[ir].shape[1] for ir in irreps_present}
        print("Irreps present with multiplicities (as dims):", dims)

    G = nx.DiGraph()
    G.add_nodes_from(irreps_present)

    for a in irreps_present:
        for b in irreps_present:
            if test_edge_activation(Qblocks, a, b, activation_fn, tol=tol):
                G.add_edge(a, b)
                if verbose:
                    print(f"edge {a} -> {b}")

    return G


# ============================================================
# 6. Visualisation (unchanged)
# ============================================================

def interaction_graph(G, title="Interaction graph inside Ind_{A4}^{A5}(1)"):
    fig, ax = plt.subplots(figsize=(7, 7))
    pos = nx.spring_layout(G, seed=42)

    bidir_edges, single_edges = [], []
    for u, v in G.edges():
        if G.has_edge(v, u) and (v, u) not in bidir_edges:
            bidir_edges.append((u, v))
        elif not G.has_edge(v, u):
            single_edges.append((u, v))

    nx.draw_networkx_nodes(G, pos, node_color="#E8F0FF", edgecolors="blue",
                           node_size=1000, linewidths=1.2)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold")
    nx.draw_networkx_edges(G, pos, edgelist=single_edges, edge_color="black",
                           arrows=True, arrowsize=25, width=1.4)
    nx.draw_networkx_edges(G, pos, edgelist=bidir_edges, edge_color="blue",
                           arrows=True, arrowsize=25, width=2.2)
    ax.set_title(title)
    ax.axis("off")
    plt.show()


def grid_interaction_graph(activation_functions, tol=1e-8, save=None):
    """
    Grid of interaction graphs for Ind_{A4}^{A5}(1).
    Rows: just one representation (so one row),
    Cols: activation functions.
    """
    num_rows = 1
    num_cols = len(activation_functions)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4))

    if num_cols == 1:
        axes = np.array([axes])

    for c, phi in enumerate(activation_functions):
        ax = axes[c]

        G = build_interaction_graph(phi, tol=tol)
        pos = nx.spring_layout(G, seed=42)

        bidir_edges, single_edges = [], []
        for u, v in G.edges():
            if G.has_edge(v, u) and (v, u) not in bidir_edges:
                bidir_edges.append((u, v))
            elif not G.has_edge(v, u):
                single_edges.append((u, v))

        nx.draw_networkx_nodes(G, pos, node_color="#E8F0FF",
                               edgecolors="blue", node_size=600, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9,
                                font_weight="bold", ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=single_edges,
                               edge_color="black", arrows=True,
                               arrowsize=12, width=1.2, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=bidir_edges,
                               edge_color="blue", arrows=True,
                               arrowsize=12, width=2.0, ax=ax)
        ax.axis("off")
        ax.set_title(phi.__name__, fontsize=12, pad=15)

    fig.suptitle(f"Induced interaction graphs", y=0.98, fontsize=14)
    plt.tight_layout(rect=[0.02, 0.02, 1, 0.92])
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()
