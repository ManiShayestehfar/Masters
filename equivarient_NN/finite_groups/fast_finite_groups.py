import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Any, Optional, Set

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sympy as sp


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _np_columnspace(M: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    if M.size == 0 or min(M.shape) == 0:
        return np.zeros((M.shape[0], 0))
    U, s, _ = np.linalg.svd(M, full_matrices=True)
    thresh = tol * s[0] if s[0] > tol else tol
    rank = int(np.sum(s > thresh))
    return np.ascontiguousarray(U[:, :rank])


def _lambdify_activation(fn: Callable) -> Callable:
    x = sp.Symbol('_x_')
    try:
        expr = fn(x)
        np_fn = sp.lambdify(
            x, expr,
            modules=[{'Max': np.maximum, 'Abs': np.abs}, 'numpy'],
        )
        _ = np.asarray(np_fn(np.array([0.5, -0.5])), dtype=float)
        return np_fn
    except Exception:
        def _fallback(v: np.ndarray) -> np.ndarray:
            return np.array(
                [float(sp.N(fn(sp.Float(float(xi))))) for xi in np.asarray(v)],
                dtype=float,
            )
        return _fallback


def generate_partitions(n: int) -> List[Tuple[int, ...]]:
    def rec(rem: int, max_part: int, current: List[int], result: List[Tuple[int, ...]]):
        if rem == 0:
            result.append(tuple(current))
            return
        for i in range(min(max_part, rem), 0, -1):
            rec(rem - i, i, current + [i], result)

    result: List[Tuple[int, ...]] = []
    rec(n, n, [], result)
    return result


def partition_label(part: Tuple[int, ...]) -> str:
    return ''.join(map(str, part))


def partition_multiplicities(part: Tuple[int, ...]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for p in part:
        out[p] = out.get(p, 0) + 1
    return out


def class_size_Sk(k: int, part: Tuple[int, ...]) -> int:
    mults = partition_multiplicities(part)
    denom = 1
    for r, m in mults.items():
        denom *= (r ** m) * math.factorial(m)
    return math.factorial(k) // denom


def representative_from_partition(k: int, part: Tuple[int, ...]) -> Tuple[int, ...]:
    perm = list(range(k))
    start = 0
    for cyc_len in part:
        if cyc_len == 1:
            start += 1
            continue
        cyc = list(range(start, start + cyc_len))
        for i in range(cyc_len):
            perm[cyc[i]] = cyc[(i + 1) % cyc_len]
        start += cyc_len
    return tuple(perm)


def point_class_sum_matrix_Sk(k: int, part: Tuple[int, ...]) -> np.ndarray:
    """
    Sum of permutation matrices of the S_k conjugacy class of cycle type `part`
    in the natural k-point permutation representation.

    By symmetry, entries depend only on whether i=j or i!=j:
      A = # {sigma in class : sigma(j)=j}
      B = # {sigma in class : sigma(j)=l} for j!=l
    """
    cls_size = class_size_Sk(k, part)
    mults = partition_multiplicities(part)
    A = 0
    if mults.get(1, 0) > 0:
        reduced = dict(mults)
        reduced[1] -= 1
        if reduced[1] == 0:
            del reduced[1]
        denom = 1
        for r, m in reduced.items():
            denom *= (r ** m) * math.factorial(m)
        A = math.factorial(k - 1) // denom
    B = (cls_size - A) // (k - 1) if k > 1 else 0

    M = np.full((k, k), float(B), dtype=np.float64)
    np.fill_diagonal(M, float(A))
    return M


def shift_matrix_Cn(n: int, c: int) -> np.ndarray:
    P = np.zeros((n, n), dtype=np.float64)
    cols = np.arange(n, dtype=np.int32)
    rows = (cols + c) % n
    P[rows, cols] = 1.0
    return P


# -----------------------------------------------------------------------------
# Lightweight class data for G = C_n x S_k, H = 1 x S_{k-1}
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FastConjugacyClass:
    index: int
    representative: Tuple[int, Tuple[int, ...]]
    partition: Tuple[int, ...]
    c_component: int
    size: int
    member_indices: frozenset


class FastCnSkModel:
    """
    Fast class/coset-action model for
        G = C_n x S_k,
        H = 1 x S_{k-1}.

    Cosets are identified with pairs (a, j), where:
      - a in C_n,
      - j is the image of k-1 under the S_k coordinate.

    Hence dim Fun(G/H) = n*k.

    The model stores only:
      - conjugacy-class data,
      - class-sum matrices on Fun(G/H),
      - irreducible character values on classes.

    It does NOT enumerate all group elements or build a Cayley table.
    """

    def __init__(self, n: int, k: int):
        self.cn = n
        self.sk = k
        self.group_order = n * math.factorial(k)
        self.cosets: List[Tuple[int, int]] = [(a, j) for a in range(n) for j in range(k)]
        self.coset_index: Dict[Tuple[int, int], int] = {x: i for i, x in enumerate(self.cosets)}
        self.dim = n * k
        self.identity = (0, tuple(range(k)))

        self.partitions = generate_partitions(k)
        self.partition_labels = [partition_label(p) for p in self.partitions]
        self.partition_to_index = {p: i for i, p in enumerate(self.partitions)}

        # S_k character table from symchar, indexed by cycle type partition.
        from symchar.symchar import character_table
        self._Sk_raw = np.array(character_table(k), dtype=object).T

        self.classes: List[FastConjugacyClass] = []
        self.class_sum_matrices: Dict[int, np.ndarray] = {}
        self._build_classes_and_class_sums()

        self.irrep_labels: List[str] = []
        self.character_table: Dict[int, List[sp.Expr]] = {}
        self._build_product_character_table()

        self._id_cls_list_idx = next(i for i, cls in enumerate(self.classes) if cls.representative == self.identity)

    def _build_classes_and_class_sums(self) -> None:
        idx = 0
        for c in range(self.cn):
            Pc = shift_matrix_Cn(self.cn, c)
            for part in self.partitions:
                rep = representative_from_partition(self.sk, part)
                size = class_size_Sk(self.sk, part)
                cls = FastConjugacyClass(
                    index=idx,
                    representative=(c, rep),
                    partition=part,
                    c_component=c,
                    size=size,
                    member_indices=frozenset({idx}),  # class-level kernels only
                )
                self.classes.append(cls)

                Ks = point_class_sum_matrix_Sk(self.sk, part)
                self.class_sum_matrices[idx] = np.kron(Pc, Ks)
                idx += 1

    def _build_product_character_table(self) -> None:
        cn_labels = [str(j) for j in range(self.cn)]
        sk_labels = self.partition_labels
        self.irrep_labels = [f"{c}⊗{s}" for c in cn_labels for s in sk_labels]

        zeta = sp.exp(2 * sp.pi * sp.I / self.cn)

        for cls in self.classes:
            part_idx = self.partition_to_index[cls.partition]
            row: List[sp.Expr] = []
            for j in range(self.cn):
                chi_c = zeta ** (j * cls.c_component)
                sk_row = self._Sk_raw[part_idx]
                for sk_val in sk_row:
                    row.append(sp.simplify(chi_c * sp.sympify(sk_val)))
            self.character_table[cls.index] = row

    def get_class_of(self, element: Tuple[int, Tuple[int, ...]]) -> FastConjugacyClass:
        if element == self.identity:
            return self.classes[self._id_cls_list_idx]
        raise NotImplementedError("Only identity-class lookup is needed in the fast solver.")


# -----------------------------------------------------------------------------
# Fast solver on class data for the above model
# -----------------------------------------------------------------------------

class FastSolver:
    """
    Fast isotypic solver for G = C_n x S_k on Fun(G/H), H = 1 x S_{k-1}.

    Supported fast path:
      - non-regular induced permutation representation on G/H,
      - isotypic projectors and isotypic interaction graphs,
      - kernel inclusion edges.

    Not implemented here:
      - generic subgroup handling,
      - full element-level group operations,
      - copy refinement / irreducible-copy interaction graphs.
    """

    def __init__(self, model: FastCnSkModel):
        self.G = model
        self.cosets = model.cosets
        self.character_table = model.character_table
        self.irrep_labels = model.irrep_labels
        self._id_cls_list_idx = model._id_cls_list_idx
        self._char_vals_np: Dict[str, np.ndarray] = {}
        self.projectors: Dict[str, np.ndarray] = {}
        self.Qblocks: Dict[str, np.ndarray] = {}

    def _compute_char_vals_np(self) -> None:
        self._char_vals_np = {
            label: np.array(
                [complex(sp.N(self.character_table[cls.index][i])) for cls in self.G.classes],
                dtype=np.complex128,
            )
            for i, label in enumerate(self.irrep_labels)
        }

    def _kernel_sets(self, labels: List[str]) -> Dict[str, Set[int]]:
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

    def compute_projectors(self) -> None:
        m = self.G.dim
        G_order = self.G.group_order

        self._compute_char_vals_np()

        complex_proj: Dict[str, np.ndarray] = {}
        for label in self.irrep_labels:
            d = self._char_vals_np[label][self._id_cls_list_idx].real
            M = np.zeros((m, m), dtype=np.complex128)
            for c_i, cls in enumerate(self.G.classes):
                M += self._char_vals_np[label][c_i].conjugate() * self.G.class_sum_matrices[cls.index]
            complex_proj[label] = (d / G_order) * M

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
                    P = 2.0 * np.real(complex_proj[label1])
                    self.projectors[label1] = P
                    self.Qblocks[label1] = _np_columnspace(P)
                    paired.add(i)

    def build_isotypic_graph(self, activation_fn: Callable) -> nx.DiGraph:
        graph = nx.DiGraph()
        act_np = _lambdify_activation(activation_fn)

        labels = [L for L in self.projectors if self.Qblocks[L].shape[1] > 0]
        for L in labels:
            graph.add_node(L, dim=self.Qblocks[L].shape[1])

        kernels = self._kernel_sets(labels)

        for src in labels:
            Q_src = self.Qblocks[src]
            for dst in labels:
                if src != dst and not kernels[src].issubset(kernels[dst]):
                    # Fast necessary-condition pruning.
                    continue
                P_dst = self.projectors[dst]
                for ci in range(Q_src.shape[1]):
                    v_act = np.asarray(act_np(Q_src[:, ci]), dtype=float)
                    if np.any(np.abs(P_dst @ v_act) > 1e-7):
                        graph.add_edge(src, dst, edge_type='activation')
                        break

        for src in labels:
            for dst in labels:
                if src != dst and not graph.has_edge(src, dst):
                    if kernels[src].issubset(kernels[dst]):
                        graph.add_edge(src, dst, edge_type='kernel')

        return graph

    def visualise_interaction_graph(
        self,
        graph: nx.DiGraph,
        node_size: int = 2500,
        font_size: int = 10,
        activation_fn: Optional[Callable] = None,
        group_name: str = "group",
        show_kernel_inclusions: bool = True,
        show_self_loops: bool = True,
        figsize: Tuple[int, int] = (10, 10),
    ) -> None:
        if len(graph.nodes) == 0:
            return

        plt.figure(figsize=figsize)

        pos = {}
        by_dim = {}
        for node, data in graph.nodes(data=True):
            dim = data.get('dim', 1)
            by_dim.setdefault(dim, []).append(node)

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

        nx.draw_networkx_nodes(graph, pos, node_color="#E8F0FF", edgecolors="#2b6cb0", node_size=node_size)
        nx.draw_networkx_labels(graph, pos, labels=display_labels, font_size=font_size, font_weight="bold")

        if kernel_edges:
            nx.draw_networkx_edges(
                graph, pos, edgelist=kernel_edges, edge_color="red", style="dotted",
                node_size=node_size, arrowsize=14, connectionstyle="arc3,rad=0.2", alpha=0.6,
            )
        if one_way_edges:
            nx.draw_networkx_edges(
                graph, pos, edgelist=one_way_edges, edge_color="black",
                node_size=node_size, arrowsize=14, connectionstyle="arc3,rad=0.0",
            )
        if mutual_edges:
            nx.draw_networkx_edges(
                graph, pos, edgelist=mutual_edges, edge_color="darkblue",
                node_size=node_size, arrowsize=14, connectionstyle="arc3,rad=0.2",
            )
        if show_self_loops and self_loops:
            nx.draw_networkx_edges(graph, pos, edgelist=self_loops, edge_color="black", node_size=node_size, arrowsize=14)

        title = group_name
        if activation_fn is not None and hasattr(activation_fn, "__name__"):
            title += f" ({activation_fn.__name__})"

        plt.title(title, fontsize=16, fontweight="bold", pad=20)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------------------------
# Convenience entry point
# -----------------------------------------------------------------------------

def run_fast_cn_sk_hkminus1(
    n: int,
    k: int,
    activation_fn: Callable,
    figsize: Tuple[int, int] = (10, 10),
    show_kernel_inclusions: bool = True,
    show_self_loops: bool = False,
):
    model = FastCnSkModel(n, k)
    solver = FastSolver(model)
    solver.compute_projectors()
    graph = solver.build_isotypic_graph(activation_fn=activation_fn)
    solver.visualise_interaction_graph(
        graph,
        activation_fn=activation_fn,
        group_name=rf"$C_{n} \times S_{k}$",
        figsize=figsize,
        show_kernel_inclusions=show_kernel_inclusions,
        show_self_loops=show_self_loops,
    )
    return graph, solver


def visualise_interaction_grid(
    graph: nx.DiGraph,
    ax,
    node_size: int = 2500,
    activation_fn=None,
    group_name: str = "group",
    show_kernel_inclusions: bool = True,
    show_self_loops: bool = False
):
    """
    Visualise interaction/isotypic graph in layered layout by dimension.

    Parameters
    ----------
    graph : nx.DiGraph
        Graph with node attribute 'dim' and edge attribute 'edge_type'
    ax : matplotlib axis
        Axis to draw on (for grid layouts)
    node_size : int
    activation_fn : callable (optional)
    group_name : str
    show_kernel_inclusions : bool
    show_self_loops : bool
    """

    if len(graph.nodes) == 0:
        ax.axis("off")
        return

    # --------------------------------------------------------
    # Layout: group nodes by dimension (top = larger dim)
    # --------------------------------------------------------
    pos = {}
    by_dim = {}

    for node, data in graph.nodes(data=True):
        dim = data.get('dim', 1)
        by_dim.setdefault(dim, []).append(node)

    sorted_dims = sorted(by_dim.keys(), reverse=True)

    for y_idx, dim in enumerate(sorted_dims):
        layer_nodes = by_dim[dim]
        n_nodes = len(layer_nodes)

        xs = np.linspace(-1, 1, n_nodes) if n_nodes > 1 else [0.0]

        for x, node in zip(xs, layer_nodes):
            pos[node] = np.array([x, -y_idx])

    # --------------------------------------------------------
    # Edge classification
    # --------------------------------------------------------
    activation_edges = [
        (u, v) for u, v, d in graph.edges(data=True)
        if d.get('edge_type', 'activation') == 'activation'
    ]

    kernel_edges = [
        (u, v) for u, v, d in graph.edges(data=True)
        if d.get('edge_type') == 'kernel'
    ]

    if not show_kernel_inclusions:
        kernel_edges = []

    self_loops = [(u, v) for u, v in activation_edges if u == v]
    mutual_edges = [(u, v) for u, v in activation_edges if u != v and (v, u) in activation_edges]
    one_way_edges = [(u, v) for u, v in activation_edges if u != v and (v, u) not in activation_edges]

    # --------------------------------------------------------
    # Draw nodes
    # --------------------------------------------------------
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color="#E8F0FF",
        edgecolors="#2b6cb0",
        node_size=node_size,
        ax=ax
    )

    # --------------------------------------------------------
    # Auto-fit labels inside nodes
    # --------------------------------------------------------
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    node_radius_pts = np.sqrt(node_size / np.pi)

    for node, (x, y) in pos.items():
        dim = graph.nodes[node].get('dim', '?')
        label = f"{node}\ndim: {dim}"

        fontsize = 1
        text = ax.text(x, y, label, ha="center", va="center", fontweight="bold")

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

    # --------------------------------------------------------
    # Draw edges
    # --------------------------------------------------------

    if kernel_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=kernel_edges,
            edge_color="red",
            style="dotted",
            node_size=node_size,
            arrowsize=14,
            connectionstyle="arc3,rad=0.2",
            alpha=0.6,
            ax=ax
        )

    if one_way_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=one_way_edges,
            edge_color="black",
            node_size=node_size,
            arrowsize=14,
            connectionstyle="arc3,rad=0.0",
            ax=ax
        )

    if mutual_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=mutual_edges,
            edge_color="darkblue",
            node_size=node_size,
            arrowsize=14,
            connectionstyle="arc3,rad=0.2",
            ax=ax
        )

    if show_self_loops and self_loops:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=self_loops,
            edge_color="black",
            node_size=node_size,
            arrowsize=14,
            ax=ax
        )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------
    title = group_name
    if activation_fn is not None and hasattr(activation_fn, "__name__"):
        title += f" ({activation_fn.__name__})"

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.axis("off")