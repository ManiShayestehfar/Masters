import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sympy as sp
from typing import Callable, Dict, List, Set, Tuple, Any, Optional

from finite_groups import FiniteGroup


# ─────────────────────────────────────────────────────────────────────────────
# Module-level numerical helpers
# ─────────────────────────────────────────────────────────────────────────────

def _np_columnspace(M: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """
    Orthonormal basis for col(M) via full SVD.
    Returns an (m, rank) array; rank == 0 gives shape (m, 0).

    This replaces SymPy's columnspace() (exact Gaussian elimination) with a
    numerically stable O(m² rank) SVD — orders of magnitude faster for m ≥ 10.
    """
    if M.size == 0 or min(M.shape) == 0:
        return np.zeros((M.shape[0], 0))
    U, s, _ = np.linalg.svd(M, full_matrices=True)
    thresh = tol * s[0] if s[0] > tol else tol
    rank = int(np.sum(s > thresh))
    return np.ascontiguousarray(U[:, :rank])


def _lambdify_activation(fn: Callable) -> Callable:
    """
    Convert a sympy scalar activation function f(x) into a vectorised numpy
    callable.  Handles sp.Max (ReLU), tanh, sigmoid, powers, and anything
    else sympy can lambdify.  Falls back to element-wise sp.N evaluation.
    """
    x = sp.Symbol('_x_')
    try:
        expr = fn(x)
        np_fn = sp.lambdify(
            x, expr,
            modules=[{'Max': np.maximum, 'Abs': np.abs}, 'numpy'],
        )
        # Smoke-test to catch mis-lambdifications early
        _ = np.asarray(np_fn(np.array([0.5, -0.5])), dtype=float)
        return np_fn
    except Exception:
        def _fallback(v: np.ndarray) -> np.ndarray:
            return np.array(
                [float(sp.N(fn(sp.Float(float(xi))))) for xi in np.asarray(v)],
                dtype=float,
            )
        return _fallback


# ─────────────────────────────────────────────────────────────────────────────

class InducedRepSolver:
    """
    Induced representation solver for G acting on G/H (permutation representation).

    All heavy linear algebra uses numpy float64 / complex128 for performance.
    Public attributes (rho_matrices, projectors, Qblocks, copy_blocks,
    copy_projectors) are numpy arrays; their .shape interface is identical to
    the former sympy matrices.

    Performance notes vs the sympy-only version:
      - Class sums use O(|G|·m) numpy scatter-adds instead of O(|G|·m²) sympy.
      - Projectors are computed with complex numpy, no sp.simplify() calls.
      - Column spaces use SVD instead of symbolic Gaussian elimination.
      - Graph nonzero checks use numpy matmul + lambdified activation functions.
    """

    def __init__(self, group: FiniteGroup):
        self.G = group
        self.cosets: List = []

        # Permutation representation ------------------------------------------
        # rho_matrices[key]   : (m, m) float64, M_g[perm[j], j] = 1
        # _rho_perms[key]     : (m,) int32  s.t. perm[j] = row index with the 1 in col j
        # _rho_inv_perms[key] : (m,) int32  s.t. (M_g @ v)[i] = v[inv_perm[i]]
        self.rho_matrices: Dict[Any, np.ndarray] = {}
        self._rho_perms: Dict[Any, np.ndarray] = {}
        self._rho_inv_perms: Dict[Any, np.ndarray] = {}

        # Character table (sympy, kept for algebraic exactness) ----------------
        self.character_table: Optional[Dict] = None
        self.irrep_labels: List[str] = []

        # Cached numpy character values (filled by _compute_char_vals_np) ------
        self._char_vals_np: Dict[str, np.ndarray] = {}  # label → (n_classes,) complex128
        self._id_cls_list_idx: int = 0                  # list-position of identity class

        # Isotypic decomposition -----------------------------------------------
        self.projectors: Dict[str, np.ndarray] = {}   # label → (m, m) float64
        self.Qblocks: Dict[str, np.ndarray] = {}      # label → (m, r) float64 orthonormal

        self.irrep_mats = None  # user-supplied dict of irrep matrices (sympy)

        # Copy decomposition ---------------------------------------------------
        self.copy_blocks: Dict[Tuple, np.ndarray] = {}     # (label, k) → (m, r) float64
        self.copy_projectors: Dict[Tuple, np.ndarray] = {} # (label, k) → (m, m) float64

    # ── subgroup / coset decomposition ───────────────────────────────────────

    def set_subgroup(self, H_elements):
        H_indices = {self.G.elem_to_idx[self.G._key(h)] for h in H_elements}
        unseen = set(range(self.G.n))
        self.cosets = []

        while unseen:
            r = unseen.pop()
            coset = {self.G.mult_table[r, h] for h in H_indices}
            self.cosets.append(frozenset(coset))
            unseen -= coset

        m = len(self.cosets)

        # O(1) element-index → coset-index lookup
        _coset_of = np.empty(self.G.n, dtype=np.int32)
        for k, C in enumerate(self.cosets):
            for x in C:
                _coset_of[x] = k

        self.rho_matrices = {}
        self._rho_perms = {}
        self._rho_inv_perms = {}
        arange_m = np.arange(m, dtype=np.int32)

        for g_idx, g in enumerate(self.G.elements):
            # perm[j] = k  iff  g maps coset j → coset k  iff  M_g[k, j] = 1
            perm = np.empty(m, dtype=np.int32)
            for j, C in enumerate(self.cosets):
                x = next(iter(C))
                perm[j] = _coset_of[self.G.mult_table[g_idx, x]]

            g_key = self.G._key(g)
            inv_perm = np.argsort(perm).astype(np.int32)
            self._rho_perms[g_key] = perm
            self._rho_inv_perms[g_key] = inv_perm   # (M_g @ v)[i] = v[inv_perm[i]]

            M = np.zeros((m, m), dtype=np.float64)
            M[perm, arange_m] = 1.0
            self.rho_matrices[g_key] = M

    # ── character table / irrep matrices ─────────────────────────────────────

    def load_character_table(self, class_character_map, labels):
        table = {}
        for cls in self.G.classes:
            table[cls.index] = [sp.sympify(x) for x in class_character_map[cls.representative]]
        self.character_table = table
        self.irrep_labels = labels

    def load_irrep_matrices(self, irrep_mats):
        self.irrep_mats = irrep_mats

    # ── numpy character cache ─────────────────────────────────────────────────

    def _compute_char_vals_np(self):
        """Convert sympy character table to complex128 numpy arrays (called once)."""
        id_cls = self.G.get_class_of(self.G.identity)
        self._id_cls_list_idx = next(
            i for i, cls in enumerate(self.G.classes) if cls.index == id_cls.index
        )
        self._char_vals_np = {
            label: np.array(
                [complex(sp.N(self.character_table[cls.index][i]))
                 for cls in self.G.classes],
                dtype=np.complex128,
            )
            for i, label in enumerate(self.irrep_labels)
        }

    # ── kernel helper ─────────────────────────────────────────────────────────

    def _kernel_sets(self, labels: List[str]) -> Dict[str, Set[int]]:
        """
        For each label return the set of group-element indices in the kernel
        of that representation.  Combined labels like 'E ⊕ E*' use the first
        component's character (kernels of ρ and ρ* coincide).
        """
        id_idx = self._id_cls_list_idx
        kernels: Dict[str, Set[int]] = {}
        for L in labels:
            if L in kernels:
                continue
            first = L.split(" ⊕ ")[0]
            cv = self._char_vals_np[first]
            chi_e = cv[id_idx]
            ker: Set[int] = set()
            for c_i, cls in enumerate(self.G.classes):
                if abs(cv[c_i] - chi_e) < 1e-7:
                    ker.update(cls.member_indices)
            kernels[L] = ker
        return kernels

    # ── isotypic projectors ───────────────────────────────────────────────────

    def compute_projectors(self, refine=False):
        m = len(self.cosets)
        G_order = self.G.n
        arange_m = np.arange(m, dtype=np.int32)

        self._compute_char_vals_np()

        # 1. Class sums (numpy scatter-add, O(|G|·m) instead of O(|G|·m²))
        #    For each g in cls:  S_cls[perm_g, arange_m] += 1
        class_sums: Dict[int, np.ndarray] = {}
        for cls in self.G.classes:
            S = np.zeros((m, m), dtype=np.float64)
            for idx in cls.member_indices:
                perm = self._rho_perms[self.G._key(self.G.elements[idx])]
                S[perm, arange_m] += 1.0
            class_sums[cls.index] = S

        # 2. Complex central idempotents  P_i = (d_i/|G|) Σ_g χ_i(g)* ρ(g)
        complex_proj: Dict[str, np.ndarray] = {}
        for i, label in enumerate(self.irrep_labels):
            d = self._char_vals_np[label][self._id_cls_list_idx].real
            M = np.zeros((m, m), dtype=np.complex128)
            for c_i, cls in enumerate(self.G.classes):
                M += self._char_vals_np[label][c_i].conjugate() * class_sums[cls.index]
            complex_proj[label] = (d / G_order) * M

        # 3. Pair conjugate characters → real projectors and Q-blocks
        self.projectors = {}
        self.Qblocks = {}
        paired: Set[int] = set()

        for i, label1 in enumerate(self.irrep_labels):
            if i in paired:
                continue
            cv1 = self._char_vals_np[label1]
            is_real = np.all(np.abs(cv1.imag) < 1e-10)

            if is_real:
                P = np.real(complex_proj[label1])
                self.projectors[label1] = P
                self.Qblocks[label1] = _np_columnspace(P)
                paired.add(i)
            else:
                conj_cv1 = cv1.conjugate()
                matched_i, matched_label = -1, None
                for j, label2 in enumerate(self.irrep_labels):
                    if j in paired or j == i:
                        continue
                    if np.allclose(conj_cv1, self._char_vals_np[label2], atol=1e-10):
                        matched_i, matched_label = j, label2
                        break

                if matched_i != -1:
                    combined = f"{label1} ⊕ {matched_label}"
                    P = np.real(complex_proj[label1] + complex_proj[matched_label])
                    self.projectors[combined] = P
                    self.Qblocks[combined] = _np_columnspace(P)
                    paired.add(i)
                    paired.add(matched_i)
                else:
                    # Fallback: doubled real part (incomplete character table)
                    P = 2.0 * np.real(complex_proj[label1])
                    self.projectors[label1] = P
                    self.Qblocks[label1] = _np_columnspace(P)
                    paired.add(i)

        if refine:
            self._refine_irreducible_copies()

    # ── copy refinement ───────────────────────────────────────────────────────

    def _refine_irreducible_copies(self):
        if self.irrep_mats is None:
            raise RuntimeError("Irrep matrices required for refinement.")

        m = len(self.cosets)
        G_order = self.G.n
        arange_m = np.arange(m, dtype=np.int32)

        # 1. Primitive idempotents  E11_α = (d/|G|) Σ_g [ρ_α(g⁻¹)]₀₀ · M_g
        #    Only the (0,0) matrix entry is needed; evaluated numerically once.
        complex_E11: Dict[str, np.ndarray] = {}
        for i, label in enumerate(self.irrep_labels):
            d = self._char_vals_np[label][self._id_cls_list_idx].real
            rho_alpha = self.irrep_mats[label]

            # Precompute all (0,0) coefficients as complex numbers (|G| sp.N calls)
            rho_alpha_np = {g: complex(sp.N(rho_alpha[g][0,0])) for g in rho_alpha}

            coeffs = np.array([
                rho_alpha_np[self.G._key(self.G.elements[self.G.inv_table[g_idx]])]
                for g_idx in range(self.G.n)
            ], dtype=np.complex128)

            # coeffs = np.array([
            #     complex(sp.N(
            #         rho_alpha[self.G._key(self.G.elements[self.G.inv_table[g_idx]])][0, 0]
            #     ))
            #     for g_idx in range(self.G.n)
            # ], dtype=np.complex128)

            E11 = np.zeros((m, m), dtype=np.complex128)
            for g_idx, g in enumerate(self.G.elements):
                perm = self._rho_perms[self.G._key(g)]
                E11[perm, arange_m] += coeffs[g_idx]

            complex_E11[label] = (d / G_order) * E11

        # 2. Same pairing logic as compute_projectors
        self.copy_blocks = {}
        self.copy_projectors = {}
        paired: Set[int] = set()

        for i, label1 in enumerate(self.irrep_labels):
            if i in paired:
                continue
            cv1 = self._char_vals_np[label1]
            is_real = np.all(np.abs(cv1.imag) < 1e-10)

            if is_real:
                if self.Qblocks.get(label1, np.zeros((m, 0))).shape[1] == 0:
                    paired.add(i)
                    continue
                self._build_copies(label1, np.real(complex_E11[label1]))
                paired.add(i)
            else:
                conj_cv1 = cv1.conjugate()
                matched_i, matched_label = -1, None
                for j, label2 in enumerate(self.irrep_labels):
                    if j in paired or j == i:
                        continue
                    if np.allclose(conj_cv1, self._char_vals_np[label2], atol=1e-10):
                        matched_i, matched_label = j, label2
                        break

                if matched_i != -1:
                    combined = f"{label1} ⊕ {matched_label}"
                    if self.Qblocks.get(combined, np.zeros((m, 0))).shape[1] == 0:
                        paired.add(i)
                        paired.add(matched_i)
                        continue
                    E11_real = np.real(complex_E11[label1] + complex_E11[matched_label])
                    self._build_copies(combined, E11_real)
                    paired.add(i)
                    paired.add(matched_i)
                else:
                    if self.Qblocks.get(label1, np.zeros((m, 0))).shape[1] == 0:
                        paired.add(i)
                        continue
                    self._build_copies(label1, 2.0 * np.real(complex_E11[label1]))
                    paired.add(i)

    def _build_copies(self, label: str, E11_real: np.ndarray):
        """
        Orbit-basis subspaces and orthogonal projectors from a real primitive
        idempotent.  For each seed vector v in col(E11):
          - Collect the G-orbit {M_g @ v : g ∈ G} using O(|G|·m) index ops.
          - Find its span via SVD.
          - Store the orthonormal basis W and projector W Wᵀ.
        """
        mult_basis = _np_columnspace(E11_real)   # (m, mult)

        for k in range(mult_basis.shape[1]):
            v = mult_basis[:, k]

            # (M_g @ v)[i] = v[inv_perm_g[i]]  — no explicit matrix needed
            orbit_cols = np.empty((v.shape[0], self.G.n), dtype=np.float64)
            for g_i, g in enumerate(self.G.elements):
                orbit_cols[:, g_i] = v[self._rho_inv_perms[self.G._key(g)]]

            W = _np_columnspace(orbit_cols)
            self.copy_blocks[(label, k)] = W
            self.copy_projectors[(label, k)] = W @ W.T   # W orthonormal ⟹ P = W Wᵀ

    # ── graph building ────────────────────────────────────────────────────────

    def build_isotypic_graph(self, activation_fn: Callable) -> nx.DiGraph:
        graph = nx.DiGraph()
        act_np = _lambdify_activation(activation_fn)

        labels = [L for L in self.projectors if self.Qblocks[L].shape[1] > 0]
        for L in labels:
            graph.add_node(L, dim=self.Qblocks[L].shape[1])

        kernels = self._kernel_sets(labels)

        # Activation edges: does σ(v) have nonzero component in dst subspace?
        for src in labels:
            Q_src = self.Qblocks[src]          # (m, r_src) orthonormal
            for dst in labels:
                P_dst = self.projectors[dst]   # (m, m) projector
                for ci in range(Q_src.shape[1]):
                    v_act = np.asarray(act_np(Q_src[:, ci]), dtype=float)
                    if np.any(np.abs(P_dst @ v_act) > 1e-7):
                        graph.add_edge(src, dst, edge_type='activation')
                        break

        # Kernel-inclusion edges
        for src in labels:
            for dst in labels:
                if src != dst and not graph.has_edge(src, dst):
                    if kernels[src].issubset(kernels[dst]):
                        graph.add_edge(src, dst, edge_type='kernel')

        return graph

    def build_interaction_graph(self, activation_fn: Callable) -> nx.DiGraph:
        graph = nx.DiGraph()
        act_np = _lambdify_activation(activation_fn)

        nodes = [key for key, W in self.copy_blocks.items() if W.shape[1] > 0]
        node_labels: Dict[Tuple, str] = {}
        for key in nodes:
            s = f"{key[0]} ({key[1] + 1})"
            node_labels[key] = s
            graph.add_node(s, dim=self.copy_blocks[key].shape[1])

        # Kernels: keyed by isotypic label (shared across copies of same irrep)
        isotypic_labels = list({key[0] for key in nodes})
        iso_kernels = self._kernel_sets(isotypic_labels)
        kernels: Dict[str, Set[int]] = {node_labels[key]: iso_kernels[key[0]] for key in nodes}

        # Activation edges
        for src_key in nodes:
            src_str = node_labels[src_key]
            W_src = self.copy_blocks[src_key]
            for dst_key in nodes:
                dst_str = node_labels[dst_key]
                P_dst = self.copy_projectors[dst_key]
                for ci in range(W_src.shape[1]):
                    v_act = np.asarray(act_np(W_src[:, ci]), dtype=float)
                    if np.any(np.abs(P_dst @ v_act) > 1e-7):
                        graph.add_edge(src_str, dst_str, edge_type='activation')
                        break

        # Kernel-inclusion edges
        for src_str in graph.nodes:
            for dst_str in graph.nodes:
                if src_str != dst_str and not graph.has_edge(src_str, dst_str):
                    if kernels[src_str].issubset(kernels[dst_str]):
                        graph.add_edge(src_str, dst_str, edge_type='kernel')

        return graph

    # ── visualisation ─────────────────────────────────────────────────────────

    def visualise_interaction_graph(
        self,
        graph: nx.DiGraph,
        node_size: int = 2500,
        font_size: int = 10,
        activation_fn: Callable = None,
        group_name: str = "group",
        show_kernel_inclusions: bool = True,
        show_self_loops: bool = True,
        figsize: Tuple[int, int] = (10, 10),
    ):
        if len(graph.nodes) > 0:
            plt.figure(figsize=figsize)

            pos = {}
            by_dim = {}
            for node, data in graph.nodes(data=True):
                dim = data.get('dim', 1)
                if dim not in by_dim:
                    by_dim[dim] = []
                by_dim[dim].append(node)

            sorted_dims = sorted(by_dim.keys(), reverse=True)
            for y_idx, dim in enumerate(sorted_dims):
                layer_nodes = by_dim[dim]
                n_nodes = len(layer_nodes)
                xs = np.linspace(-1, 1, n_nodes) if n_nodes > 1 else [0.0]
                for x, node in zip(xs, layer_nodes):
                    pos[node] = np.array([x, -y_idx])

            display_labels = {n: f"{n}\ndim: {graph.nodes[n].get('dim', '?')}" for n in graph.nodes}

            activation_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('edge_type', 'activation') == 'activation']
            kernel_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('edge_type') == 'kernel']

            if not show_kernel_inclusions:
                kernel_edges = []

            self_loops = [(u, v) for u, v in activation_edges if u == v]
            mutual_edges = [(u, v) for u, v in activation_edges if u != v and (v, u) in activation_edges]
            one_way_edges = [(u, v) for u, v in activation_edges if u != v and (v, u) not in activation_edges]

            nx.draw_networkx_nodes(
                graph, pos,
                node_color="#E8F0FF", edgecolors="#2b6cb0", node_size=node_size,
            )

            nx.draw_networkx_labels(
                graph, pos,
                labels=display_labels,
                font_size=font_size, font_weight="bold"
            )

            if kernel_edges:
                nx.draw_networkx_edges(
                    graph, pos, edgelist=kernel_edges, edge_color="red",
                    style="dotted", node_size=node_size, arrowsize=14,
                    connectionstyle="arc3,rad=0.2", alpha=0.6
                )
            if one_way_edges:
                nx.draw_networkx_edges(
                    graph, pos, edgelist=one_way_edges, edge_color="black",
                    node_size=node_size, arrowsize=14, connectionstyle="arc3,rad=0.0"
                )
            if mutual_edges:
                # Draw each directed edge separately with curvature so that
                # u→v and v→u form distinct arcs (rad=0.2 means curve left of the
                # travel direction, which is opposite for opposite-direction edges).
                nx.draw_networkx_edges(
                    graph, pos, edgelist=mutual_edges, edge_color="darkblue",
                    node_size=node_size, arrowsize=14, connectionstyle="arc3,rad=0.2",
                )
            if show_self_loops and self_loops:
                nx.draw_networkx_edges(
                    graph, pos, edgelist=self_loops, edge_color="black",
                    node_size=node_size, arrowsize=14
                )

            title = group_name
            if activation_fn is not None and hasattr(activation_fn, "__name__"):
                title += f" ({activation_fn.__name__})"

            plt.title(title, fontsize=16, fontweight="bold", pad=20)
            plt.axis("off")
            plt.tight_layout()
            plt.show()

    def visualise_interaction_grid(
        self,
        graph: nx.DiGraph,
        ax,
        node_size: int = 2500,
        activation_fn=None,
        group_name: str = "group",
        show_kernel_inclusions: bool = True,
        show_self_loops: bool = False
    ):
        if len(graph.nodes) == 0:
            ax.axis("off")
            return

        pos = {}
        by_dim = {}
        for node, data in graph.nodes(data=True):
            dim = data.get('dim', 1)
            if dim not in by_dim:
                by_dim[dim] = []
            by_dim[dim].append(node)

        sorted_dims = sorted(by_dim.keys(), reverse=True)
        for y_idx, dim in enumerate(sorted_dims):
            layer_nodes = by_dim[dim]
            n_nodes = len(layer_nodes)
            xs = np.linspace(-1, 1, n_nodes) if n_nodes > 1 else [0.0]
            for x, node in zip(xs, layer_nodes):
                pos[node] = np.array([x, -y_idx])

        activation_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('edge_type', 'activation') == 'activation']
        kernel_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('edge_type') == 'kernel']

        if not show_kernel_inclusions:
            kernel_edges = []

        self_loops = [(u, v) for u, v in activation_edges if u == v]
        mutual_edges = [(u, v) for u, v in activation_edges if u != v and (v, u) in activation_edges]
        one_way_edges = [(u, v) for u, v in activation_edges if u != v and (v, u) not in activation_edges]

        nx.draw_networkx_nodes(
            graph, pos,
            node_color="#E8F0FF", edgecolors="#2b6cb0", node_size=node_size, ax=ax
        )

        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        node_radius_pts = np.sqrt(node_size / np.pi)

        for node, (x, y) in pos.items():
            dim = graph.nodes[node].get('dim', '?')
            label = f"{node}\ndim: {dim}"

            fontsize = 1
            text = ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, fontweight="bold")

            while True:
                text.set_fontsize(fontsize)
                bbox = text.get_window_extent(renderer=renderer)
                width_pts = bbox.width * 72 / fig.dpi
                height_pts = bbox.height * 72 / fig.dpi

                if max(width_pts, height_pts) > 2 * node_radius_pts * 0.9:
                    fontsize -= 1
                    text.set_fontsize(fontsize)
                    break
                fontsize += 1

        if kernel_edges:
            nx.draw_networkx_edges(
                graph, pos, edgelist=kernel_edges, edge_color="red", style="dotted",
                node_size=node_size, arrowsize=14, connectionstyle="arc3,rad=0.2",
                alpha=0.6, ax=ax
            )
        if one_way_edges:
            nx.draw_networkx_edges(
                graph, pos, edgelist=one_way_edges, edge_color="black",
                node_size=node_size, arrowsize=14, connectionstyle="arc3,rad=0.0", ax=ax
            )
        if mutual_edges:
            # Draw each directed edge separately with curvature so that
            # u→v and v→u form distinct arcs (rad=0.2 means curve left of the
            # travel direction, which is opposite for opposite-direction edges).
            nx.draw_networkx_edges(
                graph, pos, edgelist=mutual_edges, edge_color="darkblue",
                node_size=node_size, arrowsize=14, connectionstyle="arc3,rad=0.2", ax=ax
            )
        if show_self_loops and self_loops:
            nx.draw_networkx_edges(
                graph, pos, edgelist=self_loops, edge_color="black",
                node_size=node_size, arrowsize=14, ax=ax
            )

        title = group_name
        if activation_fn is not None and hasattr(activation_fn, "__name__"):
            title += f" ({activation_fn.__name__})"

        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.axis("off")
