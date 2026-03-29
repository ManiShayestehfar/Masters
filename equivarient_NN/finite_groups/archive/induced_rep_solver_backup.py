
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

    Supports:
        - coset action construction
        - central idempotent projectors
        - optional multiplicity refinement using primitive idempotents
        - interaction graph under nonlinear activation
    """

    # ============================================================
    # INITIALISATION
    # ============================================================

    def __init__(self, group: FiniteGroup):
        self.G = group

        self.cosets = []
        self.rho_matrices = {}  # rho(g) on G/H

        self.character_table = None
        self.irrep_labels = []
        self.projectors = {}      # central idempotents
        self.Qblocks = {}         # bases of isotypic components

        self.irrep_mats = None    # optional: explicit irreps for refinement
        self.copy_blocks = {}
        self.copy_projectors = {}

    # ============================================================
    # BUILD INDUCED PERMUTATION REPRESENTATION
    # ============================================================

    def set_subgroup(self, H_elements):

        H_indices = {
            self.G.elem_to_idx[self.G._key(h)]
            for h in H_elements
        }

        unseen = set(range(self.G.n))
        self.cosets = []

        while unseen:
            r = unseen.pop()
            coset = {
                self.G.mult_table[r, h]
                for h in H_indices
            }
            self.cosets.append(frozenset(coset))
            unseen -= coset

        m = len(self.cosets)

        # build rho(g)
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

    # ============================================================
    # CHARACTER TABLE LOADING
    # ============================================================

    def load_character_table(self, class_character_map, labels):

        table = {}
        for cls in self.G.classes:
            rep_idx = self.G.elem_to_idx[self.G._key(cls.representative)]
            table[cls.index] = [
                sp.sympify(x)
                for x in class_character_map[cls.representative]
            ]

        self.character_table = table
        self.irrep_labels = labels

    # ============================================================
    # LOAD EXPLICIT IRREP MATRICES (OPTIONAL)
    # ============================================================

    def load_irrep_matrices(self, irrep_mats):
        self.irrep_mats = irrep_mats

    # ============================================================
    # CENTRAL IDEMPOTENTS
    # ============================================================

    def compute_projectors(self, refine=False):

        m = len(self.cosets)
        G_order = sp.Integer(self.G.n)

        # class sums
        class_sums = {}
        for cls in self.G.classes:
            S = sp.zeros(m)
            for idx in cls.member_indices:
                g = self.G.elements[idx]
                S += self.rho_matrices[self.G._key(g)]
            class_sums[cls.index] = S

        # central idempotents
        id_class = self.G.get_class_of(self.G.identity)
        dims = self.character_table[id_class.index]

        self.projectors = {}
        self.Qblocks = {}

        for i, label in enumerate(self.irrep_labels):

            d = dims[i]

            M = sp.zeros(m)
            for cls in self.G.classes:
                chi = self.character_table[cls.index][i]
                M += sp.conjugate(chi) * class_sums[cls.index]

            P = sp.simplify((d / G_order) * M)
            self.projectors[label] = P

            # basis of isotypic component
            basis = P.columnspace()
            self.Qblocks[label] = (
                sp.Matrix.hstack(*basis) if basis else sp.Matrix()
            )

        if refine:
            self._refine_irreducible_copies()

    # ============================================================
    # OPTIONAL: PRIMITIVE IDEMPOTENTS
    # ============================================================

    def _refine_irreducible_copies(self):

        if self.irrep_mats is None:
            raise RuntimeError("Irrep matrices required for refinement.")

        m = len(self.cosets)
        G_order = sp.Integer(self.G.n)

        self.copy_blocks = {}
        self.copy_projectors = {}

        id_class = self.G.get_class_of(self.G.identity)
        dims = self.character_table[id_class.index]

        for i, label in enumerate(self.irrep_labels):

            d = int(dims[i])
            P = self.projectors[label]

            if P.rank() == 0:
                continue

            rho_alpha = self.irrep_mats[label]

            # primitive idempotent E_11
            E11 = sp.zeros(m)

            for g in self.G.elements:
                g_key = self.G._key(g)
                g_idx = self.G.elem_to_idx[g_key]
                g_inv = self.G.elements[self.G.inv_table[g_idx]]

                coeff = rho_alpha[self.G._key(g_inv)][0, 0]
                E11 += coeff * self.rho_matrices[g_key]

            E11 = sp.simplify((d / G_order) * E11)

            mult_space = E11.columnspace()
            mult = len(mult_space)

            for k in range(mult):

                v = mult_space[k]

                orbit = []
                for g in self.G.elements:
                    orbit.append(
                        self.rho_matrices[self.G._key(g)] * v
                    )

                W = sp.Matrix.hstack(*orbit).columnspace()
                Wmat = sp.Matrix.hstack(*W)

                self.copy_blocks[(label, k)] = Wmat
                self.copy_projectors[(label, k)] = (
                    Wmat * (Wmat.T * Wmat).inv() * Wmat.T
                )

    # ============================================================
    # INTERACTION GRAPH
    # ============================================================

    def build_interaction_graph(self, activation_fn):

        graph = nx.DiGraph()

        labels = [
            L for L in self.Qblocks
            if self.Qblocks[L].shape[1] > 0
        ]

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
                    v_act = v.applyfunc(activation_fn)
                    proj = P * v_act

                    if nonzero(proj):
                        graph.add_edge(src, dst)
                        break

        return graph