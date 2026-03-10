import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sympy as sp
from dataclasses import dataclass
from typing import List, Callable, Dict, Any, Hashable, Tuple, Optional, Set, Union

from finite_groups import FiniteGroup

class InducedRepSolver:
    """
    Induced representation solver for:
        G acts on G/H (permutation representation)
    """

    def __init__(self, group: FiniteGroup):
        self.G = group
        self.cosets = []
        self.rho_matrices = {}  
        self.character_table = None
        self.irrep_labels = []
        self.projectors = {}      
        self.Qblocks = {}         
        self.irrep_mats = None    
        self.copy_blocks = {}
        self.copy_projectors = {}

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

        self.rho_matrices = {}
        for g_idx, g in enumerate(self.G.elements):
            M = sp.zeros(m)
            for j, C in enumerate(self.cosets):
                x = next(iter(C))
                gx = self.G.mult_table[g_idx, x]
                for k, D in enumerate(self.cosets):
                    if gx in D:
                        M[k, j] = 1
                        break
            self.rho_matrices[self.G._key(g)] = M

    def load_character_table(self, class_character_map, labels):
        table = {}
        for cls in self.G.classes:
            rep_idx = self.G.elem_to_idx[self.G._key(cls.representative)]
            table[cls.index] = [sp.sympify(x) for x in class_character_map[cls.representative]]
        self.character_table = table
        self.irrep_labels = labels

    def load_irrep_matrices(self, irrep_mats):
        self.irrep_mats = irrep_mats

    def compute_projectors(self, refine=False):
        m = len(self.cosets)
        G_order = sp.Integer(self.G.n)

        # 1. Compute class sums
        class_sums = {}
        for cls in self.G.classes:
            S = sp.zeros(m)
            for idx in cls.member_indices:
                g = self.G.elements[idx]
                S += self.rho_matrices[self.G._key(g)]
            class_sums[cls.index] = S

        id_class = self.G.get_class_of(self.G.identity)
        dims = self.character_table[id_class.index]

        # 2. Compute ALL complex central idempotents
        complex_projectors = {}
        for i, label in enumerate(self.irrep_labels):
            d = dims[i]
            M = sp.zeros(m)
            for cls in self.G.classes:
                chi = self.character_table[cls.index][i]
                M += sp.conjugate(chi) * class_sums[cls.index]
            P = sp.simplify((d / G_order) * M)
            complex_projectors[label] = P

        # 3. Pair up conjugate characters to form unified real projectors
        self.projectors = {}
        self.Qblocks = {}
        paired_indices = set()
        
        for i, label1 in enumerate(self.irrep_labels):
            if i in paired_indices:
                continue
                
            char1 = [self.character_table[cls.index][i] for cls in self.G.classes]
            
            # Check if character is strictly real
            is_real = all(sp.simplify(sp.im(c)) == 0 for c in char1)
            
            if is_real:
                # Real representation
                P_real = sp.simplify(complex_projectors[label1].applyfunc(sp.re))
                self.projectors[label1] = P_real
                basis = P_real.columnspace()
                self.Qblocks[label1] = sp.Matrix.hstack(*basis) if basis else sp.Matrix()
                paired_indices.add(i)
            else:
                # Find conjugate pair
                conj_char1 = [sp.conjugate(c) for c in char1]
                matched_idx = -1
                matched_label = None
                
                for j, label2 in enumerate(self.irrep_labels):
                    if j in paired_indices or i == j:
                        continue
                    char2 = [self.character_table[cls.index][j] for cls in self.G.classes]
                    
                    # Check if char2 is the exact complex conjugate of char1
                    if all(sp.N(c1 - c2, chop=True) == 0 for c1, c2 in zip(conj_char1, char2)):
                        matched_idx = j
                        matched_label = label2
                        break
                        
                if matched_idx != -1:
                    # Merge P1 and P2
                    combined_label = f"{label1} ⊕ {matched_label}"
                    P_sum = complex_projectors[label1] + complex_projectors[matched_label]
                    P_real = sp.simplify(P_sum.applyfunc(sp.re))
                    
                    self.projectors[combined_label] = P_real
                    basis = P_real.columnspace()
                    self.Qblocks[combined_label] = sp.Matrix.hstack(*basis) if basis else sp.Matrix()
                    
                    paired_indices.add(i)
                    paired_indices.add(matched_idx)
                else:
                    # Fallback (only triggered if character table is incomplete/missing the conjugate pair)
                    P_real = sp.simplify(2 * complex_projectors[label1].applyfunc(sp.re))
                    self.projectors[label1] = P_real
                    basis = P_real.columnspace()
                    self.Qblocks[label1] = sp.Matrix.hstack(*basis) if basis else sp.Matrix()
                    paired_indices.add(i)

        if refine:
            self._refine_irreducible_copies()

    def _refine_irreducible_copies(self):
        if self.irrep_mats is None:
            raise RuntimeError("Irrep matrices required for refinement.")

        m = len(self.cosets)
        G_order = sp.Integer(self.G.n)

        self.copy_blocks = {}
        self.copy_projectors = {}

        id_class = self.G.get_class_of(self.G.identity)
        dims = self.character_table[id_class.index]

        # 1. Compute ALL complex primitive idempotents E11
        complex_E11 = {}
        for i, label in enumerate(self.irrep_labels):
            d = int(dims[i])
            rho_alpha = self.irrep_mats[label]

            E11 = sp.zeros(m)
            for g in self.G.elements:
                g_key = self.G._key(g)
                g_idx = self.G.elem_to_idx[g_key]
                g_inv = self.G.elements[self.G.inv_table[g_idx]]

                coeff = rho_alpha[self.G._key(g_inv)][0, 0]
                E11 += coeff * self.rho_matrices[g_key]

            E11 = sp.simplify((d / G_order) * E11)
            complex_E11[label] = E11

        # 2. Pair them up using the same character logic as the central idempotents
        paired_indices = set()
        for i, label1 in enumerate(self.irrep_labels):
            if i in paired_indices:
                continue
                
            char1 = [self.character_table[cls.index][i] for cls in self.G.classes]
            is_real = all(sp.simplify(c - sp.conjugate(c)) == 0 for c in char1)  
                      
            if is_real:
                # Real representation
                if self.projectors[label1].rank() == 0:
                    paired_indices.add(i)
                    continue
                    
                E11_real = sp.simplify(complex_E11[label1].applyfunc(sp.re))
                self._build_copies(label1, E11_real)
                paired_indices.add(i)
            else:
                # Find conjugate pair
                conj_char1 = [sp.conjugate(c) for c in char1]
                matched_idx = -1
                matched_label = None
                
                for j, label2 in enumerate(self.irrep_labels):
                    if j in paired_indices or i == j:
                        continue
                    char2 = [self.character_table[cls.index][j] for cls in self.G.classes]
                    
                    if all(sp.simplify(c1 - c2) == 0 for c1, c2 in zip(conj_char1, char2)):
                        matched_idx = j
                        matched_label = label2
                        break
                        
                if matched_idx != -1:
                    combined_label = f"{label1} ⊕ {matched_label}"
                    if self.projectors[combined_label].rank() == 0:
                        paired_indices.add(i)
                        paired_indices.add(matched_idx)
                        continue
                        
                    E11_sum = complex_E11[label1] + complex_E11[matched_label]
                    E11_real = sp.simplify(E11_sum.applyfunc(sp.re))
                    
                    self._build_copies(combined_label, E11_real)
                    
                    paired_indices.add(i)
                    paired_indices.add(matched_idx)
                else:
                    # Fallback
                    if self.projectors[label1].rank() == 0:
                        paired_indices.add(i)
                        continue
                    E11_real = sp.simplify(2 * complex_E11[label1].applyfunc(sp.re))
                    self._build_copies(label1, E11_real)
                    paired_indices.add(i)

    def _build_copies(self, label, E11_real):
        """Helper to compute orbit basis spaces and sub-projectors from real primitive idempotents."""
        mult_space = E11_real.columnspace()
        mult = len(mult_space)

        for k in range(mult):
            v = mult_space[k]
            orbit = []
            for g in self.G.elements:
                orbit.append(self.rho_matrices[self.G._key(g)] * v)

            W = sp.Matrix.hstack(*orbit).columnspace()
            Wmat = sp.Matrix.hstack(*W)

            self.copy_blocks[(label, k)] = Wmat
            self.copy_projectors[(label, k)] = (
                Wmat * (Wmat.T * Wmat).inv() * Wmat.T
            )

    def build_isotypic_graph(self, activation_fn):
        graph = nx.DiGraph()

        labels = [
            L for L in self.projectors
            if self.Qblocks[L].shape[1] > 0
        ]

        # NEW: Store the dimension as a node attribute
        for L in labels:
            dim = self.Qblocks[L].shape[1]
            graph.add_node(L, dim=dim)

        def nonzero(M):
            return any(abs(sp.N(x)) > 1e-7 for x in M)

        for src in labels:
            P_src = self.projectors[src]
            basis = P_src.columnspace()
            for dst in labels:
                P_dst = self.projectors[dst]
                for v in basis:
                    v_act = v.applyfunc(activation_fn)
                    if nonzero(P_dst * v_act):
                        graph.add_edge(src, dst, edge_type='activation')
                        break

        kernels = {}
        for L in labels:
            first_irrep_label = L.split(" ⊕ ")[0]
            idx = self.irrep_labels.index(first_irrep_label)
            ker_L = set()
            id_class = self.G.get_class_of(self.G.identity)
            chi_e = self.character_table[id_class.index][idx]
            for cls in self.G.classes:
                chi_g = self.character_table[cls.index][idx]
                if abs(sp.N(chi_g - chi_e)) < 1e-7:
                    for member_idx in cls.member_indices:
                        ker_L.add(member_idx)
            kernels[L] = ker_L

        for src in labels:
            for dst in labels:
                if src != dst and not graph.has_edge(src, dst):
                    if kernels[src].issubset(kernels[dst]):
                        graph.add_edge(src, dst, edge_type='kernel')

        return graph

    def build_interaction_graph(self, activation_fn):
        graph = nx.DiGraph()
        
        nodes = [
            node_key for node_key in self.copy_blocks 
            if self.copy_blocks[node_key].shape[1] > 0
        ]
        
        node_labels = {}
        # NEW: Store the dimension as a node attribute
        for node_key in nodes:
            dim = self.copy_blocks[node_key].shape[1]
            node_str = f"{node_key[0]} ({node_key[1] + 1})"
            node_labels[node_key] = node_str
            graph.add_node(node_str, dim=dim)

        def nonzero(M):
            return any(abs(sp.N(x)) > 1e-7 for x in M)

        for src_key in nodes:
            src_str = node_labels[src_key]
            Q_src = self.copy_blocks[src_key]
            for dst_key in nodes:
                dst_str = node_labels[dst_key]
                P_dst = self.copy_projectors[dst_key]
                for j in range(Q_src.shape[1]):
                    v = Q_src.col(j)
                    v_act = v.applyfunc(activation_fn)
                    proj = P_dst * v_act
                    if nonzero(proj):
                        graph.add_edge(src_str, dst_str, edge_type='activation')
                        break

        kernels = {}
        for node_key in nodes:
            label = node_key[0]
            first_irrep_label = label.split(" ⊕ ")[0]
            idx = self.irrep_labels.index(first_irrep_label)
            ker = set()
            id_class = self.G.get_class_of(self.G.identity)
            chi_e = self.character_table[id_class.index][idx]
            for cls in self.G.classes:
                chi_g = self.character_table[cls.index][idx]
                if abs(sp.N(chi_g - chi_e)) < 1e-7:
                    for member_idx in cls.member_indices:
                        ker.add(member_idx)
            kernels[node_labels[node_key]] = ker

        for src_str in graph.nodes:
            for dst_str in graph.nodes:
                if src_str != dst_str and not graph.has_edge(src_str, dst_str):
                    if kernels[src_str].issubset(kernels[dst_str]):
                        graph.add_edge(src_str, dst_str, edge_type='kernel')

        return graph
    

    def visualise_interaction_graph(
        self, 
        graph: nx.DiGraph, 
        node_size: int = 2500, 
        font_size: int = 10, 
        activation_fn: Callable = None, 
        group_name: str = "C3xS4", 
        show_kernel_inclusions: bool = True,
        show_self_loops: bool = True # NEW: Toggle for self-loops
    ):
        if len(graph.nodes) > 0:
            plt.figure(figsize=(10, 10))
            
            # Compute Hasse diagram layout from high to low dimension
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
            
            # Straightened kernel edges (rad=0.0)
            if kernel_edges:
                nx.draw_networkx_edges(
                    graph, pos, edgelist=kernel_edges, edge_color="red", 
                    style="dotted", node_size=node_size, arrowsize=14,
                    connectionstyle="arc3,rad=0.0", alpha=0.6 
                )
            # Straightened one-way edges (rad=0.0)
            if one_way_edges:
                nx.draw_networkx_edges(
                    graph, pos, edgelist=one_way_edges, edge_color="black", 
                    node_size=node_size, arrowsize=14, connectionstyle="arc3,rad=0.0"
                )
            # Mutual edges retain a tiny curve (rad=0.1) so both directions are visible
            if mutual_edges:
                nx.draw_networkx_edges(
                    graph, pos, edgelist=mutual_edges, edge_color="darkblue", 
                    node_size=node_size, arrowsize=14, connectionstyle="arc3,rad=0.1"
                )
            # Self-loops conditionally rendered
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
        group_name: str = "C3xS4", 
        show_kernel_inclusions: bool = True,
        show_self_loops: bool = False # NEW: Toggle for self-loops
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

        # Straightened kernel edges
        if kernel_edges:
            nx.draw_networkx_edges(
                graph, pos, edgelist=kernel_edges, edge_color="red", style="dotted",
                node_size=node_size, arrowsize=14, connectionstyle="arc3,rad=0.0",
                alpha=0.6, ax=ax
            )

        # Straightened one-way edges
        if one_way_edges:
            nx.draw_networkx_edges(
                graph, pos, edgelist=one_way_edges, edge_color="black",
                node_size=node_size, arrowsize=14, connectionstyle="arc3,rad=0.0", ax=ax
            )

        # Mutual edges retain a tiny curve
        if mutual_edges:
            nx.draw_networkx_edges(
                graph, pos, edgelist=mutual_edges, edge_color="darkblue",
                node_size=node_size, arrowsize=14, connectionstyle="arc3,rad=0.1", ax=ax
            )

        # Conditionally render self-loops
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