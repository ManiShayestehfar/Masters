import math
import itertools as it
import numpy as np
import scipy.linalg
import scipy.special
import networkx as nx
import matplotlib.pyplot as plt

# ============================================================
# 1. Fast Combinatorics & Johnson Scheme Logic
# ============================================================

def get_johnson_matrix(n, k):
    """
    Constructs the Adjacency Matrix of the Johnson Graph J(n, k).
    Nodes are k-subsets. Edges exist if intersection size is k-1.
    
    Complexity: O(M^2 * n) where M = nCk.
    This avoids iterating S_n (n!) entirely.
    """
    # 1. Generate all k-subsets as a boolean matrix
    # shape: (M, n), where M = n choose k
    subsets_list = list(it.combinations(range(n), k))
    m = len(subsets_list)
    
    # Create a binary matrix representation of subsets
    # X[i, j] = 1 if element j is in subset i
    X = np.zeros((m, n), dtype=np.int8)
    for row_idx, subset in enumerate(subsets_list):
        X[row_idx, list(subset)] = 1

    # 2. Compute Intersection Matrix via matrix multiplication
    # I[i, j] = |Subset_i ∩ Subset_j|
    intersection_matrix = X @ X.T
    
    # 3. The Adjacency Matrix A of J(n, k)
    # Nodes are connected if they differ by exactly 1 element 
    # (intersection size == k - 1)
    A = (intersection_matrix == (k - 1)).astype(float)
    
    return A, subsets_list

def get_irreps_spectral(n, k, tol=1e-8):
    """
    Recovers the irreducible subspaces (Q blocks) by diagonalizing 
    the Johnson Graph adjacency matrix.
    
    The eigenvalues of J(n,k) for irrep (n-i, i) are known explicitly:
    λ_i = (k - i)(n - k - i) - i
    """
    if k < 0 or k > n:
        return {}, [], []

    # Build adjacency
    A, basis = get_johnson_matrix(n, k)
    
    # Eigendecomposition (Hermitian/Symmetric is faster)
    # eigenvalues w are returned in ascending order
    w, v = scipy.linalg.eigh(A)
    
    # Flip to descending order to match theoretical index i=0,1,...
    w = w[::-1]
    v = v[:, ::-1]
    
    Q_dict = {}
    
    # Theoretical eigenvalues for i = 0 to k
    # λ_i = (k-i)(n-k-i) - i
    # Since these are strictly decreasing with i, we can match them by sorting.
    
    current_col = 0
    for i in range(k + 1):
        # Calculate theoretical dimension
        dim_i = math.comb(n, i) - math.comb(n, i - 1) if i > 0 else 1
        
        # Extract the eigenvectors corresponding to this block
        # Because standard solvers might have slight numerical drift, 
        # we slice by expected dimension rather than approximate float comparison 
        # (though strictly, w[current_col] should ≈ theoretical_lambda)
        
        if dim_i > 0:
            Q_i = v[:, current_col : current_col + dim_i]
            Q_dict[i] = Q_i
            current_col += dim_i
        else:
            Q_dict[i] = np.zeros((len(basis), 0))

    # Q_full is just the concatenation (which is v, sorted)
    Q_full = v
    
    return Q_dict, Q_full, basis

# ============================================================
# 2. Interaction Graph (Optimized)
# ============================================================

def test_edge_activation_fast(Qi, Qj, activation_fn, tol=1e-8):
    """
    Vectorized leak test. 
    Checks if Qj.T @ activation(Qi @ x) is non-zero.
    We test using the basis vectors of the subspace Qi.
    """
    if Qi.size == 0 or Qj.size == 0:
        return False
        
    # Instead of searching for a specific vector v, we project the 
    # entire basis of Qi through the nonlinearity.
    # This is a heuristic: if the subspace maps to the other subspace,
    # it usually happens for the basis vectors.
    
    # Project standard basis of Qi through activation
    # Input shape: (M, dim_i)
    
    # 1. Apply activation to the columns of Qi (the basis of the subspace)
    # This checks if basis vectors trigger the leak.
    act_basis = activation_fn(Qi) 
    
    # 2. Project onto subspace Qj
    # Result shape: (dim_j, dim_i)
    projection = Qj.T @ act_basis
    
    # Check magnitude
    return np.any(np.abs(projection) > tol)

def build_interaction_graph(n, k, activation_fn, tol=1e-8, verbose=False):
    if verbose:
        print(f"Building graph for n={n}, k={k}")
    
    # Fast spectral construction
    Qblocks, _, _ = get_irreps_spectral(n, k, tol)
    
    irreps = sorted([i for i, Q in Qblocks.items() if Q.shape[1] > 0])
    
    G = nx.DiGraph()
    for i in irreps:
        # Calculate dimension: (n choose i) - (n choose i-1)
        dim_i = math.comb(n, i) - (math.comb(n, i-1) if i > 0 else 0)
        
        # Add node with attribute
        node_id = (n-i, i)
        G.add_node(node_id, dimension=dim_i)
    
    # Only apply activation once per block to speed up checks
    # (Assuming activation_fn is element-wise numpy callable)
    
    for i in irreps:
        for j in irreps:
            if test_edge_activation_fast(Qblocks[i], Qblocks[j], activation_fn, tol):
                G.add_edge((n-i, i), (n-j, j))
                if verbose:
                    print(f"Edge {(n-i,i)} -> {(n-j,j)}")
    return G

# ============================================================
# 3. Visualization (Preserved & Cleaned)
# ============================================================

def grid_interaction_graph(N_list, k_list, activation_functions, tol=1e-6, save=None):
    num_rows = len(N_list)
    num_cols = len(activation_functions)
    
    # Store adjacency matrices here
    adj_matrices = {}
    
    # Dynamic figsize
    fig, axes = plt.subplots(num_rows, num_cols, 
                             figsize=(3.5 * num_cols, 3.5 * num_rows),
                             squeeze=False) 

    for r, (n, k) in enumerate(zip(N_list, k_list)):
        for c, phi in enumerate(activation_functions):
            ax = axes[r, c]
            
            # Build
            G = build_interaction_graph(n, k, phi, tol=tol)
            
            # --- Capture Adjacency Matrix ---
            # We strictly order nodes by their partition index 'i' (the second element of the tuple)
            # This ensures rows/cols follow the sequence (n,0), (n-1,1), ..., (n-k,k)
            sorted_nodes = sorted(G.nodes(), key=lambda x: x[1])
            adj = nx.to_numpy_array(G, nodelist=sorted_nodes)
            adj_matrices[(n, k, phi.__name__)] = adj
            
            # Layout
            pos = nx.spring_layout(G, seed=42, k=2.0) # k=2.0 spreads nodes out more
            
            # Edges
            bidir = []
            single = []
            for u, v in G.edges():
                if G.has_edge(v, u) and (v, u) not in bidir:
                    bidir.append((u, v))
                elif not G.has_edge(v, u):
                    single.append((u, v))

            # Draw
            nx.draw_networkx_nodes(G, pos, node_color="#E8F0FF", edgecolors="#3b82f6", 
                                   node_size=1000, ax=ax)
            
            # Create labels with Dimension included
            labels = {}
            for node in G.nodes(data=True):
                node_id, attrs = node
                dim = attrs.get('dimension', '?')
                # Format: Partition on top, Dim below
                # node_id is (n-i, i). We can just show the 'i' or the full tuple.
                labels[node_id] = f"{node_id}\n$d={dim}$"
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, 
                                    font_weight="bold", ax=ax)
                                    
            nx.draw_networkx_edges(G, pos, edgelist=single, edge_color="#334155", 
                                   arrows=True, arrowsize=15, width=1.5, ax=ax)
            nx.draw_networkx_edges(G, pos, edgelist=bidir, edge_color="#3b82f6", 
                                   arrows=True, arrowsize=15, width=2.0, connectionstyle='arc3,rad=0.1', ax=ax)
            
            ax.axis("off")
            
            if r == 0:
                ax.set_title(phi.__name__, fontsize=11, weight='bold')
        
        # Row label
        fig.text(0.02, (num_rows - r - 0.5)/num_rows, f"N={n}\nk={k}", 
                 va='center', ha='center', fontsize=10, weight='bold', rotation=90)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
    return adj_matrices

def plot_hyperplane(Q, activation_fn, a_range=(-3,3), b_range=(-3,3), grid_size=100):
    """Visualization helper (vectorized)"""
    if Q.shape[0] != 3:
        print("Warning: Can only visualise 3D slices.")
        return None, None, None

    Q = np.array(Q, dtype=float)
    # Use pseudo-inverse for stability if not perfectly orthogonal
    Qinv = np.linalg.pinv(Q) 

    a = np.linspace(a_range[0], a_range[1], grid_size)
    b = np.linspace(b_range[0], b_range[1], grid_size)
    A, B = np.meshgrid(a, b)
    
    # Flatten: (N_pixels, 3) => (0, a, b)
    flat_in = np.zeros((A.size, 3))
    flat_in[:, 1] = A.ravel()
    flat_in[:, 2] = B.ravel()
    
    # Forward: x -> Qx -> phi(Qx)
    # Q.T maps from coefficients to standard basis
    projected = flat_in @ Q.T 
    activated = activation_fn(projected)
    
    # Backward: phi(Qx) -> Q^-1 phi(Qx)
    recovered = activated @ Qinv.T
    
    # We want the component along index 0
    F = recovered[:, 0].reshape(A.shape)
    
    return A, B, F

def hyperplane_contour(Q, activation_fn, title="Nonlinear section"):
    A, B, F = plot_hyperplane(Q, activation_fn)
    if A is None: return

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(A, B, F, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_title(title)
    ax1.set_xlabel("a (dim 1)")
    ax1.set_ylabel("b (dim 2)")
    
    ax2 = fig.add_subplot(1, 2, 2)
    cont = ax2.contourf(A, B, F, levels=25, cmap='viridis')
    plt.colorbar(cont, ax=ax2)
    ax2.set_title("Contour Map")
    
    plt.tight_layout()
    plt.show()

