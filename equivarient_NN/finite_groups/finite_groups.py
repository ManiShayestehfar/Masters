import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sympy as sp
from dataclasses import dataclass
from typing import List, Callable, Dict, Any, Hashable, Tuple, Optional, Set, Union

# ============================================================
# 1. Generic Finite Group Class
# ============================================================

@dataclass
class ConjugacyClass:
    """
    Stores info about a conjugacy class.
    """
    index: int
    representative: Any
    members: List[Any]
    member_indices: Set[int] # Indices for robust lookup
    size: int

class FiniteGroup:
    """
    A class to represent any finite group G.
    Handles unhashable elements (like numpy arrays or mutable SymPy matrices) 
    by generating internal keys.
    """
    def __init__(self, elements: List[Any], mult_func: Callable[[Any, Any], Any], 
                 classes: Optional[List[Tuple[Any, List[Any]]]] = None):
        self.elements = elements
        self.mult_func = mult_func
        self.n = len(elements)
        
        # 1. Setup Key Strategy for Hashability
        if self.n > 0:
            sample = elements[0]
            # Handle NumPy Arrays
            if isinstance(sample, np.ndarray):
                # Fix: Use flattened tuple instead of tobytes().
                # We check hasattr(x, 'ravel') to safely handle cases where mixed types 
                # (e.g., tuples) might be passed despite the sample being an array.
                self._key = lambda x: tuple(x.ravel()) if hasattr(x, 'ravel') else tuple(x)
            # Handle SymPy Mutable Matrices
            elif hasattr(sample, "as_immutable"): 
                self._key = lambda x: x.as_immutable()
            else:
                try:
                    hash(sample)
                    self._key = lambda x: x
                except TypeError:
                    # Fallback: try converting to tuple
                    self._key = lambda x: tuple(x)
        else:
            self._key = lambda x: x

        # 2. Create mapping from key(element) -> index
        self.elem_to_idx = {self._key(e): i for i, e in enumerate(elements)}
        
        # 3. Build Cayley Table (Multiplication Table)
        self.mult_table = np.zeros((self.n, self.n), dtype=int)
        
        print(f"Building multiplication table for G (order {self.n})...")
        for i, e1 in enumerate(elements):
            for j, e2 in enumerate(elements):
                prod = mult_func(e1, e2)
                k_prod = self._key(prod)
                if k_prod not in self.elem_to_idx:
                    # Provide helpful debug info
                    raise ValueError(f"Closure violation at indices ({i},{j}). \n"
                                     f"Product:\n{prod}\n"
                                     f"is not in the provided element list (or hash key mismatch).")
                self.mult_table[i, j] = self.elem_to_idx[k_prod]

        # 4. Find Identity
        self.identity_idx = -1
        for i in range(self.n):
            if self.mult_table[i, i] == i:
                is_identity = True
                for k in range(self.n):
                    if self.mult_table[i, k] != k or self.mult_table[k, i] != k:
                        is_identity = False
                        break
                if is_identity:
                    self.identity_idx = i
                    break
        
        if self.identity_idx == -1:
            raise ValueError("No identity element found in the group.")

        self.identity = self.elements[self.identity_idx]

        # 5. Build Inverse Map
        self.inv_table = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            found = False
            for k in range(self.n):
                if self.mult_table[i, k] == self.identity_idx:
                    self.inv_table[i] = k
                    found = True
                    break
            if not found:
                raise ValueError(f"Element {elements[i]} has no inverse.")
        
        # 6. Conjugacy Classes
        if classes is not None:
            print("Using user-provided Conjugacy Classes.")
            self.classes = []
            for idx, (rep, members) in enumerate(classes):
                member_indices = set()
                valid_members = []
                for m in members:
                    k_m = self._key(m)
                    if k_m not in self.elem_to_idx:
                        raise ValueError(f"Class member {m} not found in group elements.")
                    m_idx = self.elem_to_idx[k_m]
                    member_indices.add(m_idx)
                    valid_members.append(m)
                
                self.classes.append(ConjugacyClass(
                    index=idx,
                    representative=rep,
                    members=valid_members,
                    member_indices=member_indices,
                    size=len(valid_members)
                ))
        else:
            self.classes = self._discover_conjugacy_classes()
        
    def _discover_conjugacy_classes(self) -> List[ConjugacyClass]:
        """Compute conjugacy classes by brute force orbit calculation on indices."""
        print("Discovering conjugacy classes...")
        unseen = set(range(self.n))
        classes = []
        class_idx = 0
        
        while unseen:
            g_idx = unseen.pop()
            orb_indices = set()
            
            # Compute orbit of g under conjugation by all h in G
            for h_idx in range(self.n):
                h_inv_idx = self.inv_table[h_idx]
                step1 = self.mult_table[h_idx, g_idx]
                conj = self.mult_table[step1, h_inv_idx]
                orb_indices.add(conj)
            
            unseen -= orb_indices
            members_indices = sorted(list(orb_indices))
            members = [self.elements[i] for i in members_indices]
            rep = members[0] 
            
            c = ConjugacyClass(
                index=class_idx,
                representative=rep,
                members=members,
                member_indices=set(members_indices),
                size=len(members)
            )
            classes.append(c)
            class_idx += 1
            
        return classes

    def get_class_of(self, element: Any) -> ConjugacyClass:
        k = self._key(element)
        if k not in self.elem_to_idx:
            raise ValueError("Element not in group")
        idx = self.elem_to_idx[k]
        for c in self.classes:
            if idx in c.member_indices:
                return c
        raise RuntimeError(f"Element {element} not found in any class.")


# ============================================================
# 2. Induced Representation Solver (SymPy Enabled)
# ============================================================

class InducedRepSolver:
    """
    Manages the subgroup H, the cosets G/H, the induced representation matrices,
    and the interaction graph generation using SymPy for exact arithmetic.
    """
    def __init__(self, group: FiniteGroup):
        self.G = group
        self.H_elements = []
        self.cosets = []  # List of frozensets (integers)
        self.coset_reps = [] 
        self.rho_matrices = {} # Map element_key -> SymPy Matrix
        self.projectors = {}
        self.Qblocks = {}
        self.character_table = None 
        self.irrep_labels = []

    def set_subgroup(self, h_elements: List[Any]):
        
        self.H_elements = h_elements
        
        H_indices = set()
        for h in h_elements:
            k = self.G._key(h)
            if k not in self.G.elem_to_idx:
                raise ValueError(f"Subgroup element {h} not in G.")
            H_indices.add(self.G.elem_to_idx[k])
        
        unseen = set(range(self.G.n))
        self.cosets = []
        self.coset_reps = []
        
        while unseen:
            r_idx = unseen.pop()
            coset_indices = set()
            for h_idx in H_indices:
                val = self.G.mult_table[r_idx, h_idx]
                coset_indices.add(val)
            
            self.cosets.append(frozenset(coset_indices))
            self.coset_reps.append(self.G.elements[r_idx])
            unseen -= coset_indices

        m = len(self.cosets)
        print(f"Computed {m} cosets for H (order {len(h_elements)}) in G (order {self.G.n}).")
        
        # Precompute Permutation Action Matrices rho(g) on G/H as SymPy Matrices
        self.rho_matrices = {}
        
        for g_idx in range(self.G.n):
            g = self.G.elements[g_idx]
            g_key = self.G._key(g)
            
            # Create exact SymPy matrix
            mat = sp.zeros(m, m)
            
            for j, c_indices in enumerate(self.cosets):
                x_idx = next(iter(c_indices))
                gx_idx = self.G.mult_table[g_idx, x_idx]
                
                target_idx = -1
                for k, target_c_indices in enumerate(self.cosets):
                    if gx_idx in target_c_indices:
                        target_idx = k
                        break
                
                if target_idx == -1:
                    raise RuntimeError("Coset action failed.")
                mat[target_idx, j] = 1 # Exact integer
            
            self.rho_matrices[g_key] = mat

    def load_character_table(self, mapping: Union[Dict[Any, List[Any]], List[Tuple[Any, List[Any]]]], 
                           irrep_labels: List[str] = None):
        """
        Load character table using SymPy expressions.
        """
        index_to_vector = {}
        items = list(mapping.items()) if isinstance(mapping, dict) else mapping
            
        for rep, vec in items:
            k = self.G._key(rep)
            if k not in self.G.elem_to_idx:
                 raise ValueError(f"Representative {rep} not found in Group.")
            idx = self.G.elem_to_idx[k]
            # Convert to SymPy expressions
            index_to_vector[idx] = [sp.sympify(x) for x in vec]

        class_vectors = {}
        num_irreps = 0
        
        for cls in self.G.classes:
            found_vec = None
            rep_idx = self.G.elem_to_idx[self.G._key(cls.representative)]
            
            if rep_idx in index_to_vector:
                found_vec = index_to_vector[rep_idx]
            else:
                for m_idx in cls.member_indices:
                    if m_idx in index_to_vector:
                        found_vec = index_to_vector[m_idx]
                        break
            
            if found_vec is None:
                raise ValueError(f"No character vector provided for Class containing {cls.representative}.")
            
            if num_irreps == 0:
                num_irreps = len(found_vec)
            elif len(found_vec) != num_irreps:
                raise ValueError(f"Inconsistent lengths. Expected {num_irreps}, got {len(found_vec)}.")
                
            class_vectors[cls.index] = found_vec

        self.character_table = class_vectors
        self.num_irreps = num_irreps
        self.irrep_labels = irrep_labels or [f"Irrep_{i}" for i in range(num_irreps)]
        print("Character table loaded (SymPy).")

    def compute_projectors(self):
        """
        Compute exact projectors using SymPy.
        """
        if not self.rho_matrices or not self.character_table:
            raise RuntimeError("Subgroup or Character Table not set.")

        m = len(self.cosets)
        group_order = sp.Integer(self.G.n)
        
        # Precompute Class Sums (Sum of SymPy Matrices)
        class_sums = {}
        for cls in self.G.classes:
            S = sp.zeros(m, m)
            for g_idx in cls.member_indices:
                g = self.G.elements[g_idx]
                k = self.G._key(g)
                S += self.rho_matrices[k]
            class_sums[cls.index] = S

        self.projectors = {}
        self.Qblocks = {}
        
        for alpha_idx in range(self.num_irreps):
            label = self.irrep_labels[alpha_idx]
            
            # Dimension d_alpha
            id_class = self.G.get_class_of(self.G.identity)
            chi_vec = self.character_table[id_class.index]
            d_alpha = chi_vec[alpha_idx]
            
            M = sp.zeros(m, m)
            for cls in self.G.classes:
                val = self.character_table[cls.index][alpha_idx]
                val_conj = sp.conjugate(val)
                M += val_conj * class_sums[cls.index]
            
            P = (d_alpha / group_order) * M
            
            self.projectors[label] = P
            
            # Exact Eigendecomposition to find subspace for lambda=1
            eigendata = P.eigenvects()
            
            basis_vectors = []
            for (eval_sym, mult, vecs) in eigendata:
                # Check if eigenvalue is 1
                if eval_sym == 1:
                    basis_vectors.extend(vecs)
                elif (eval_sym - 1).simplify() == 0:
                     basis_vectors.extend(vecs)
            
            if basis_vectors:
                self.Qblocks[label] = sp.Matrix.hstack(*basis_vectors)
            else:
                self.Qblocks[label] = sp.Matrix() # Empty

    def build_interaction_graph(self, activation_fn, verbose=False) -> nx.DiGraph:
        """
        Build interaction graph using exact symbolic checks.
        activation_fn must handle SymPy expressions (e.g., use sp.Max(0, x)).
        """
        present_labels = [L for L, Q in self.Qblocks.items() if Q.shape[1] > 0]
        
        if verbose:
            print("Irreps present:", present_labels)

        G_graph = nx.DiGraph()
        G_graph.add_nodes_from(present_labels)
        
        def test_edge(L_src, L_dst):
            Q_src = self.Qblocks[L_src]
            Q_dst = self.Qblocks[L_dst]
            
            # Check single basis vectors
            for a in range(Q_src.shape[1]):
                v = Q_src.col(a)
                v_act = v.applyfunc(activation_fn)
                
                P_dst = self.projectors[L_dst]
                projected = P_dst * v_act
                
                if not projected.is_zero_matrix:
                     return True
            
            # Check sums of pairs
            for a in range(Q_src.shape[1]):
                for b in range(a+1, Q_src.shape[1]):
                    v = Q_src.col(a) + Q_src.col(b)
                    v_act = v.applyfunc(activation_fn)
                    projected = self.projectors[L_dst] * v_act
                    if not projected.is_zero_matrix:
                        return True
            return False

        for src in present_labels:
            for dst in present_labels:
                if test_edge(src, dst):
                    G_graph.add_edge(src, dst)
                    
        return G_graph