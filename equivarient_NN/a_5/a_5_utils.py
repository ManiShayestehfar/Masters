import itertools as it
from collections import defaultdict
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

def cycle_type(p):
    """Cycle type partition of permutation p."""
    n = len(p)
    seen = [False]*n
    cyc = []
    for i in range(n):
        if not seen[i]:
            j = i
            L = 0
            while not seen[j]:
                seen[j] = True
                j = p[j]
                L += 1
            cyc.append(L)
    cyc.sort(reverse=True)
    return tuple(cyc)

def order_of_perm(p):
    """Order from cycle type."""
    o = 1
    for L in cycle_type(p):
        o = np.lcm(o, L)
    return o


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
    Then label them as 1A, 2A, 3A, 5A, 5B using standard reps.

    Facts (for sanity checking):
      classes of 1, (123), (12)(34), (12345), (12354)
      have sizes 1, 20, 15, 12, 12.  
    """
    G = A5_elements()
    Gset = set(G)

    unseen = set(G)
    classes = []
    while unseen:
        g = unseen.pop()
        # conjugacy orbit under A5
        orb = set()
        for h in G:
            c = compose(compose(h, g), inv(h))
            orb.add(c)
        unseen -= orb
        classes.append(list(orb))

    # pick a representative for each class (canonical: min tuple)
    reps = [min(C) for C in classes]

    # build lookup rep -> members
    rep_to_members = {rep: sorted(C) for rep, C in zip(reps, classes)}

    # label using containment of standard reps
    e = tuple(range(5))
    rep_3 = (1,2,0,3,4)          # (0 1 2)
    rep_22 = (1,0,3,2,4)         # (0 1)(2 3)
    rep_5A = (1,2,3,4,0)         # (0 1 2 3 4)
    rep_5B = inv(rep_5A)         # (0 4 3 2 1)

    # find which computed class contains each standard rep
    def find_class_containing(x):
        for rep, members in rep_to_members.items():
            if x in members:
                return rep
        raise RuntimeError("Class not found")

    rep_1A = find_class_containing(e)
    rep_3A = find_class_containing(rep_3)
    rep_2A = find_class_containing(rep_22)
    rep_5a = find_class_containing(rep_5A)
    rep_5b = find_class_containing(rep_5B)

    labelled = [
        A5Class("1A", rep_1A, rep_to_members[rep_1A]),
        A5Class("3A", rep_3A, rep_to_members[rep_3A]),
        A5Class("2A", rep_2A, rep_to_members[rep_2A]),
        A5Class("5A", rep_5a, rep_to_members[rep_5a]),
        A5Class("5B", rep_5b, rep_to_members[rep_5b]),
    ]

    # sanity check sizes (1,20,15,12,12) :contentReference[oaicite:3]{index=3}
    sizes = {C.label: len(C.members) for C in labelled}
    if sorted(sizes.values()) != sorted([1, 20, 15, 12, 12]):
        raise RuntimeError(f"Unexpected class sizes: {sizes}")

    return labelled


# ============================================================
# 2. The representation: A5 acting on k-subsets of {0..4}
#    (restriction of the S5 permutation representation)
# ============================================================

def k_subsets_5(k):
    """k-subsets of {0,1,2,3,4} as sorted tuples."""
    if k < 0 or k > 5:
        return []
    return [tuple(c) for c in it.combinations(range(5), k)]

def perm_matrix_on_k_subsets(k, sigma):
    """
    Permutation matrix of sigma on k-subsets (basis ordered by k_subsets_5).
    """
    Xk = k_subsets_5(k)
    idx = {A: i for i, A in enumerate(Xk)}
    m = len(Xk)
    M = np.zeros((m, m), dtype=float)
    for j, A in enumerate(Xk):
        B = tuple(sorted(sigma[a] for a in A))
        M[idx[B], j] = 1.0
    return M


# ============================================================
# 3. Character table of A5 (standard labelling)
# ============================================================

def character_table():
    """
    Returns dict: irreps -> dict(class_label -> character_value).
    Irreps labelled by their dimension: "1", "3", "3p", "4", "5".

    Values are the standard A5 character table.
    From Conrad handout:
      χ4: (4, 1, 0, -1, -1) on (1A,3A,2A,5A,5B)
      χ5: (5, -1, 1, 0, 0)                             
      χ3, χ3' have x=0 on 3-cycles, y=-1 on 2A, and
        z,w = (1±√5)/2 on the two 5-cycle classes.      
    """
    sqrt5 = np.sqrt(5.0)
    phi  = (1.0 + sqrt5)/2.0
    phi2 = (1.0 - sqrt5)/2.0

    # class order: 1A, 3A, 2A, 5A, 5B
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
# ============================================================

def projectors_on_Vk(k, tol=1e-10):
    """
    Build projectors p_irrep on V_k = R[X_k] where X_k = k-subsets of {0..4},
    for the restricted A5-action.

      p_χ = (d_χ/|A5|) Σ_{g∈A5} χ(g^{-1}) ρ(g).

    Since the A5 characters above are real-valued, χ(g^{-1}) = χ(g),
    but we keep the inverse to match the general formula.

    Returns:
      proj   : dict irrep_label -> (m×m) numpy matrix
      Qblocks: dict irrep_label -> (m×r) ONB columns for image(proj)
      basis  : list of k-subsets indexing the standard basis
      classes: computed A5 conjugacy classes (labelled)
    """
    basis = k_subsets_5(k)
    m = len(basis)
    G = A5_elements()
    classes = conjugacy_classes()
    chartab = character_table()
    group_order = 60

    # Precompute rho(g)
    rho = {g: perm_matrix_on_k_subsets(k, g) for g in G}

    # Precompute class sums S_C = Σ_{g∈C} rho(g)
    class_sums = {}
    rep_to_label = {}
    for C in classes:
        rep_to_label[C.rep] = C.label
        S = np.zeros((m, m), dtype=float)
        for g in C.members:
            S += rho[g]
        class_sums[C.label] = S

    proj = {}
    Qblocks = {}

    for ir in chartab.keys():
        d = irrep_dimension(ir)
        M = np.zeros((m, m), dtype=float)

        for C in classes:
            lab = C.label
            # character at class (use inverse in the formula if you generalise)
            chi = float(chartab[ir][lab])
            M += chi * class_sums[lab]

        p = (d / group_order) * M
        p = 0.5 * (p + p.T)  # symmetrise (helps numerically)
        proj[ir] = p

        # Extract an ONB for image(p): eigenvalue ~ 1
        vals, vecs = np.linalg.eigh(p)
        idx1 = np.where(np.abs(vals - 1.0) < tol)[0]
        Q = vecs[:, idx1]
        Qblocks[ir] = Q

    return proj, Qblocks, basis, classes


# ============================================================
# 5. Interaction graph (same as your S_n version, but on A5 blocks)
# ============================================================

def test_edge_activation(Qblocks, ir_a, ir_b, activation_fn, tol=1e-8):
    """
    Edge ir_a -> ir_b exists if there is v in span(Q_ir_a)
    such that (apply activation elementwise) has nonzero projection
    to span(Q_ir_b).
    """
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

    # optional: pairwise sums (often increases sensitivity)
    for a in range(Qa.shape[1]):
        for b in range(a+1, Qa.shape[1]):
            if leaks(Qa[:, a] + Qa[:, b]):
                return True

    return False


def build_interaction_graph(k, activation_fn, tol=1e-8, verbose=False):
    """
    Interaction graph inside V_k under A5-action on k-subsets of {0..4}.
    Nodes: irreps of A5: 1, 3, 3', 4, 5.
    """
    _, Qblocks, _, _ = projectors_on_Vk(k)

    irreps_present = [ir for ir, Q in Qblocks.items() if Q.shape[1] > 0]

    G = nx.DiGraph()
    G.add_nodes_from(irreps_present)

    for a in irreps_present:
        for b in irreps_present:
            if test_edge_activation(Qblocks, a, b, activation_fn, tol=tol):
                G.add_edge(a, b)
                if verbose:
                    print(f"edge {a} -> {b}")

    return G

def interaction_graph(G, title="Interaction graph inside V_{5,k}"):
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


def grid_interaction_graph(k_list, activation_functions, tol=1e-8, save=None):
    """
    Grid of interaction graphs.
    Rows correspond to (N,k) pairs.
    Columns correspond to activation functions.
    Each row is labeled on the left with 'N = ..., k = ...'
    """
    num_rows = len(k_list)
    num_cols = len(activation_functions)
    fig, axes = plt.subplots(num_rows, num_cols,
                             figsize=(4 * num_cols, 4 * num_rows))
    
    # handle single-row/column cases gracefully
    if num_rows == 1:
        axes = np.array([axes])
    if num_cols == 1:
        axes = axes.reshape(-1, 1)

    for r, (n, k) in enumerate(zip(5*np.ones(len(k_list), dtype=int), k_list)):
        for c, phi in enumerate(activation_functions):
            ax = axes[r, c]

            # --- Build the interaction graph ---
            G = build_interaction_graph(k, phi, tol=tol)
            pos = nx.spring_layout(G, seed=42)

            # --- Split edges into single/bidirectional ---
            bidir_edges, single_edges = [], []
            for u, v in G.edges():
                if G.has_edge(v, u) and (v, u) not in bidir_edges:
                    bidir_edges.append((u, v))
                elif not G.has_edge(v, u):
                    single_edges.append((u, v))

            # --- Draw graph ---
            nx.draw_networkx_nodes(G, 
                                   pos, node_color="#E8F0FF",
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

            # --- Column and row labels ---
            if r == 0:
                ax.set_title(phi.__name__, fontsize=12, pad=15)

        # Add LaTeX-styled text label for the row (N,k)
        fig.text(
            0.01,                                # x position (near left)
            (num_rows - r - 0.7) / num_rows,     # y position per row
            rf"$N={n},\,k={k}$",                 # LaTeX label
            ha="left", va="center", fontsize=13
        )

    plt.tight_layout(rect=[0.15, 0.08, 1, 0.95])  # left, bottom, right, top
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()