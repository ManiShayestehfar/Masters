import math
import itertools as it
from itertools import combinations
from collections import Counter

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ============================================================
# 1. Combinatorial helpers
# ============================================================

def integer_partitions(n):
    """Generate integer partitions of n in non-increasing order."""
    def gen(n, max_part):
        if n == 0:
            yield ()
            return
        for x in range(min(n, max_part), 0, -1):
            for rest in gen(n - x, x):
                yield (x,) + rest
    return list(gen(n, n))


def class_size_from_type(n, part):
    """|C_μ| = n! / (Π_i (m_i! * i^{m_i}))"""
    mult = Counter(part)
    denom = 1
    for i, m in mult.items():
        denom *= math.factorial(m) * (i**m)
    return math.factorial(n) // denom


def all_perms(n):
    """All permutations of {0,...,n-1}."""
    return list(it.permutations(range(n)))


def cycle_type(p):
    """Cycle-type partition of permutation p."""
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


def conjugacy_classes(n):
    """Brute-force list of conjugacy classes of S_n."""
    classes = {}
    for g in all_perms(n):
        mu = cycle_type(g)
        classes.setdefault(mu, []).append(g)
    out = []
    for mu, members in classes.items():
        out.append({"type": mu, "size": len(members), "members": members})
    return out


# ============================================================
# 2. k-subset permutation representation ρ : S_n → GL(V_{n,k})
# ============================================================

def k_subsets(n, k):
    if k < 0 or k > n:
        return []
    return [tuple(c) for c in it.combinations(range(n), k)]


def perm_matrix(n, k, sigma):
    """Permutation matrix of σ acting on k-subsets."""
    Xk = k_subsets(n, k)
    idx = {A: i for i, A in enumerate(Xk)}
    m = len(Xk)
    M = np.zeros((m, m))
    for j, A in enumerate(Xk):
        B = tuple(sorted(sigma[a] for a in A))
        M[idx[B], j] = 1
    return M


# ============================================================
# 3. Two-row irreducible dimensions and characters (numeric)
# ============================================================

def dim_two_row(n, i):
    """dim S^{(n-i,i)} = C(n,i) – C(n,i-1)."""
    if i == 0:
        return 1
    if i < 0 or i > n:
        return 0
    return math.comb(n, i) - math.comb(n, i-1)


def chi_two_row(n, i, cycle_type_part):
    """
    Fast numeric version of
    χ^{(n-i,i)}(σ)
    = [x^i] ∏_{m≥1} (1+x^m)^{a_m} − [x^{i−1}]∏_{m≥1}(1+x^m)^{a_m},
    where a_m = # of m-cycles in σ.
    """
    c = Counter(cycle_type_part)
    max_power = sum(m * am for m, am in c.items())
    coeffs = np.zeros(max_power + 1, dtype=np.int64)
    coeffs[0] = 1

    for m, am in c.items():
        # Multiply by (1 + x^m)^am
        for _ in range(am):
            new_coeffs = np.zeros_like(coeffs)
            new_coeffs[m:] += coeffs[:-m]  # shift by m
            coeffs += new_coeffs
    if i == 0:
        return 1
    return coeffs[i] - coeffs[i-1]


# ============================================================
# 4. Classical character-sum projectors (numeric)
# ============================================================

def projectors_two_row_characters(n, k, tol=1e-8):
    """
    Compute projectors p_{(n−i,i)} on V_{n,k} = ℝ[X_k]
    using the character-sum formula (numeric, fast).
    """
    basis = k_subsets(n, k)
    m = len(basis)
    classes = conjugacy_classes(n)

    # Precompute permutation matrices ρ(g)
    rho = {}
    for C in classes:
        for g in C["members"]:
            rho[g] = perm_matrix(n, k, g)

    proj, Qblocks = {}, {}
    for i in range(0, k+1):
        dimL = dim_two_row(n, i)
        if dimL == 0:
            Qblocks[i] = np.zeros((m, 0))
            continue

        M = np.zeros((m, m))
        for C in classes:
            mu = C["type"]
            chi = chi_two_row(n, i, mu)
            S_C = np.sum([rho[g] for g in C["members"]], axis=0)
            M += chi * S_C

        p = (dimL / math.factorial(n)) * M
        p = (p + p.T) / 2  # enforce symmetry
        proj[i] = p

        # eigenvectors for eigenvalue ≈ 1
        vals, vecs = np.linalg.eigh(p)
        mask = np.isclose(vals, 1.0, atol=tol)
        eig1 = vecs[:, mask]
        Qblocks[i] = eig1

    return proj, Qblocks, basis


def assemble_Q(n, k, tol=1e-8, normalise=True):
    """Assemble all Q-blocks into one orthogonal matrix."""
    _, Qblocks, basis = projectors_two_row_characters(n, k, tol)
    m = len(basis)
    Q_full_cols = []
    Q_dict = {}

    for i, Q_i in Qblocks.items():
        if Q_i.shape[1] == 0:
            continue
        if normalise:
            Q_i = Q_i / np.linalg.norm(Q_i, axis=0, keepdims=True)
        Q_dict[i] = Q_i
        Q_full_cols.append(Q_i)
    Q_full = np.concatenate(Q_full_cols, axis=1) if Q_full_cols else np.zeros((m, 0))
    return Q_dict, Q_full, basis


# ============================================================
# 5. Interaction graph (numeric)
# ============================================================

def test_edge_activation(Qblocks, i, j, activation_fn, tol=1e-8):
    Qi, Qj = Qblocks.get(i), Qblocks.get(j)
    if Qi is None or Qj is None or Qi.size == 0 or Qj.size == 0:
        return False
    phi = np.vectorize(activation_fn)

    def leaks(v):
        v_act = phi(v)
        coeffs = Qj.T @ v_act
        return np.any(np.abs(coeffs) > tol)

    for a in range(Qi.shape[1]):
        if leaks(Qi[:, a]):
            return True
    return False


def build_interaction_graph(n, k, activation_fn, tol=1e-8, verbose=False):
    if verbose:
        print(f"Building graph for n={n}, k={k}")
    _, Qblocks, _ = projectors_two_row_characters(n, k, tol)
    irreps = [i for i, Q in Qblocks.items() if Q.shape[1] > 0]
    G = nx.DiGraph()
    nodes = [(n-i, i) for i in irreps]
    G.add_nodes_from(nodes)
    for i in irreps:
        for j in irreps:
            if test_edge_activation(Qblocks, i, j, activation_fn, tol):
                G.add_edge((n-i, i), (n-j, j))
                if verbose:
                    print(f"Edge {(n-i,i)}→{(n-j,j)}")
    return G


# ============================================================
# 6. Visualisation
# ============================================================

def interaction_graph(G, title="Interaction graph inside V_{n,k}"):
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

def grid_interaction_graph(N_list, k_list, activation_functions, tol=1e-8, save=None):
    """
    Grid of interaction graphs.
    Rows correspond to (N,k) pairs.
    Columns correspond to activation functions.
    Each row is labeled on the left with 'N = ..., k = ...'
    """
    num_rows = len(N_list)
    num_cols = len(activation_functions)
    fig, axes = plt.subplots(num_rows, num_cols,
                             figsize=(4 * num_cols, 4 * num_rows))
    
    # handle single-row/column cases gracefully
    if num_rows == 1:
        axes = np.array([axes])
    if num_cols == 1:
        axes = axes.reshape(-1, 1)

    for r, (n, k) in enumerate(zip(N_list, k_list)):
        for c, phi in enumerate(activation_functions):
            ax = axes[r, c]

            # --- Build the interaction graph ---
            G = build_interaction_graph(n, k, phi, tol=tol)
            pos = nx.spring_layout(G, seed=42)

            # --- Split edges into single/bidirectional ---
            bidir_edges, single_edges = [], []
            for u, v in G.edges():
                if G.has_edge(v, u) and (v, u) not in bidir_edges:
                    bidir_edges.append((u, v))
                elif not G.has_edge(v, u):
                    single_edges.append((u, v))

            # --- Draw graph ---
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

            # --- Column and row labels ---
            if r == 0:
                ax.set_title(phi.__name__, fontsize=12, pad=15)

        # Add LaTeX-styled text label for the row (N,k)
        fig.text(
            0.04,                                # x position (near left)
            (num_rows - r - 0.5) / num_rows,     # y position per row
            rf"$N={n},\,k={k}$",                 # LaTeX label
            ha="left", va="center", fontsize=13
        )

    plt.tight_layout(rect=[0.15, 0.08, 1, 0.95])  # left, bottom, right, top
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()



def plot_hyperplane(Q, activation_fn, a_range=(-3,3), b_range=(-3,3), grid_size=100):
    """
    Compute and visualize (a,b) ↦ (1,0,0)^T (Q^{-1} ◦ φ ◦ Q)(0,a,b)
    where φ is a numpy-based activation function.

    Args:
        Q : (m×m) numpy.ndarray or convertible
        activation_fn : lambda x: ...  (must accept numpy arrays)
        a_range, b_range : ranges for the grid
        grid_size : resolution of the contour plot

    Returns:
        A, B, F : meshgrids for plotting
    """
    assert Q.shape[0] == 3, "Can only visualise in 3D"

    Q = np.array(Q, dtype=float)
    Qinv = np.linalg.inv(Q)

    # grid
    a_vals = np.linspace(*a_range, grid_size)
    b_vals = np.linspace(*b_range, grid_size)
    A, B = np.meshgrid(a_vals, b_vals)

    # flatten inputs
    inputs = np.stack([np.zeros_like(A), A, B], axis=-1).reshape(-1, 3)

    # compute φ(Qv)
    inner = inputs @ Q.T                 # (N,3)
    act = activation_fn(inner)           # elementwise activation
    if act.shape != inner.shape:
        raise ValueError("activation_fn must be elementwise, returning same shape")

    # back-transform
    out = act @ Qinv.T

    # project to first coordinate
    F = out[:, 0].reshape(A.shape)
    return A, B, F


def hyperplane_contour(Q, activation_fn, title="Nonlinear section", **kwargs):
    assert Q.shape[0] == 3, "Can only visualise in 3D"
    
    A, B, F = plot_hyperplane(Q, activation_fn, **kwargs)

    fig = plt.figure(figsize=(10,4))
    
    # 3D surface
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    surf = ax1.plot_surface(A, B, F, cmap='cividis', alpha=0.9, edgecolor='none')
    ax1.set_xlabel("a")
    ax1.set_ylabel("b")
    ax1.set_zlabel("f(a,b)")
    ax1.set_title(title, fontsize=12)
    # fig.colorbar(surf, ax=ax1, shrink=0.5)

    # Contour plot
    ax2 = fig.add_subplot(1,2,2)
    cont = ax2.contourf(A, B, F, levels=20, cmap='cividis')
    ax2.contour(A, B, F, levels=20, colors='black', linewidths=0.3, alpha=0.4)
    ax2.set_xlabel("a")
    ax2.set_ylabel("b")
    # ax2.set_title("Contour of f(a,b)", fontsize=12)
    fig.colorbar(cont, ax=ax2)
    
    plt.tight_layout()
    plt.show()