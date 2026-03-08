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
                    if all(sp.simplify(c1 - c2) == 0 for c1, c2 in zip(conj_char1, char2)):
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

    def build_interaction_graph(self, activation_fn):
        graph = nx.DiGraph()
        labels = [L for L in self.Qblocks if self.Qblocks[L].shape[1] > 0]
        graph.add_nodes_from(labels)

        def nonzero(M):
            return any(sp.simplify(x) != 0 for x in M)

        for src in labels:
            for dst in labels:
                Q = self.Qblocks[src]
                P = self.projectors[dst]

                # test generic vectors in source space
                for j in range(Q.shape[1]):
                    v = Q.col(j)
                    # v is now strictly real, so activation_fn executes cleanly
                    v_act = v.applyfunc(activation_fn)
                    proj = P * v_act

                    if nonzero(proj):
                        graph.add_edge(src, dst)
                        break

        return graph
    
    def build_isotypic_graph(self, activation_fn):

        graph = nx.DiGraph()

        labels = [
            L for L in self.projectors
            if self.projectors[L].rank() > 0
        ]

        graph.add_nodes_from(labels)

        def nonzero(M):
            return any(sp.simplify(x) != 0 for x in M)

        for src in labels:
            P_src = self.projectors[src]

            # basis for V[src]
            basis = P_src.columnspace()

            for dst in labels:
                P_dst = self.projectors[dst]

                for v in basis:

                    v_act = v.applyfunc(activation_fn)

                    if nonzero(P_dst * v_act):
                        graph.add_edge(src, dst)
                        break

        return graph

    def visualise_interaction_graph(self, graph: nx.DiGraph, node_size: int = 1800, activation_fn: Callable = None, group_name: str = "C3xS4"):
        """
        Visualises the interaction graph using networkx and matplotlib.
        """
        if len(graph.nodes) > 0:
            plt.figure(figsize=(8, 8))
            
            # Using a circular layout evenly spaces the nodes, reducing jumbled intersections
            pos = nx.circular_layout(graph)
            
            # Separate edges into categories for different styling
            self_loops = [(u, v) for u, v in graph.edges() if u == v]
            mutual_edges = [(u, v) for u, v in graph.edges() if u != v and graph.has_edge(v, u)]
            one_way_edges = [(u, v) for u, v in graph.edges() if u != v and not graph.has_edge(v, u)]
            
            # Draw nodes
            nx.draw_networkx_nodes(
                graph, pos, 
                node_color="#E8F0FF", 
                edgecolors="#2b6cb0", 
                node_size=node_size,
            )
            
            # Draw labels
            nx.draw_networkx_labels(
                graph, pos, 
                font_size=12, 
                font_weight="bold"
            )
            
            # Draw one-way edges (curved, black)
            if one_way_edges:
                nx.draw_networkx_edges(
                    graph, pos, 
                    edgelist=one_way_edges, 
                    edge_color="black", 
                    node_size=1800,
                    arrowsize=18,
                    connectionstyle="arc3,rad=0.1"
                )
                
            # Draw mutual edges (straight, dark blue, overlapping creates bidirectional arrows)
            if mutual_edges:
                nx.draw_networkx_edges(
                    graph, pos, 
                    edgelist=mutual_edges, 
                    edge_color="darkblue", 
                    node_size=1800,
                    arrowsize=18,
                    connectionstyle="arc3,rad=0.0"
                )
                
            # Draw self-loops
            if self_loops:
                nx.draw_networkx_edges(
                    graph, pos, 
                    edgelist=self_loops, 
                    edge_color="black", 
                    node_size=1800,
                    arrowsize=18
                )
            
            title = group_name
            if activation_fn is not None and hasattr(activation_fn, "__name__"):
                title += f" ({activation_fn.__name__})"
                
            plt.title(title, fontsize=16, fontweight="bold", pad=20)
            plt.axis("off")  # Turn off the surrounding axes frame
            plt.tight_layout()
            plt.show()