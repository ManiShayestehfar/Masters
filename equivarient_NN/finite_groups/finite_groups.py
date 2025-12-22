import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Callable, Dict, Any, Hashable, Tuple, Optional, Set

# ============================================================
# 1. Generic Finite Group Class
# ============================================================

@dataclass(frozen=True)
class ConjugacyClass:
    index: int
    representative: Hashable
    members: List[Hashable]
    size: int

class FiniteGroup:
    """
    A class to represent any finite group G.
    It manages element indexing, multiplication tables, inverses, 
    and conjugacy class discovery.
    """
    def __init__(self, elements: List[Hashable], mult_func: Callable[[Any, Any], Any]):
        self.elements = elements
        self.mult_func = mult_func
        self.n = len(elements)
        
        # Create mapping from element -> index
        # Elements must be hashable. If not, user must wrap them (e.g. tuples).
        self.elem_to_idx = {e: i for i, e in enumerate(elements)}
        
        # 1. Build Cayley Table (Multiplication Table)
        # table[i, j] = index(elements[i] * elements[j])
        self.mult_table = np.zeros((self.n, self.n), dtype=int)
        
        # We also need to identify the Identity element (e s.t. xe = ex = x)
        # We'll find it during construction or assume closure.
        
        print(f"Building multiplication table for G (order {self.n})...")
        for i, e1 in enumerate(elements):
            for j, e2 in enumerate(elements):
                prod = mult_func(e1, e2)
                if prod not in self.elem_to_idx:
                    raise ValueError(f"Closure violation: {e1} * {e2} = {prod}, which is not in the element list.")
                self.mult_table[i, j] = self.elem_to_idx[prod]

        # 2. Find Identity
        self.identity_idx = -1
        for i in range(self.n):
            # Check if e * e = e. Necessary but not sufficient, but good heuristic start.
            if self.mult_table[i, i] == i:
                # rigorous check
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

        # 3. Build Inverse Map
        self.inv_table = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            # Find k such that i * k = identity
            found = False
            for k in range(self.n):
                if self.mult_table[i, k] == self.identity_idx:
                    self.inv_table[i] = k
                    found = True
                    break
            if not found:
                raise ValueError(f"Element {elements[i]} has no inverse.")
        
        # 4. Auto-discover Conjugacy Classes
        self.classes = self._discover_conjugacy_classes()
        
    def _discover_conjugacy_classes(self) -> List[ConjugacyClass]:
        """Compute conjugacy classes by brute force orbit calculation."""
        unseen = set(range(self.n))
        classes = []
        class_idx = 0
        
        while unseen:
            g_idx = unseen.pop()
            orb_indices = set()
            
            # Compute orbit of g under conjugation by all h in G
            # orbit = { h * g * h^-1 for h in G }
            for h_idx in range(self.n):
                h_inv_idx = self.inv_table[h_idx]
                # prod = h * g * h^-1
                # internal calc: (h * g) then * h_inv
                step1 = self.mult_table[h_idx, g_idx]
                conj = self.mult_table[step1, h_inv_idx]
                orb_indices.add(conj)
            
            # Remove found orbit from unseen
            unseen -= orb_indices
            
            # Sort members for consistency (if elements are comparable), otherwise just generic sort by index
            members_indices = sorted(list(orb_indices))
            members = [self.elements[i] for i in members_indices]
            
            # Representative is the one with the smallest index (usually first found or 'smallest' value)
            rep = members[0] 
            
            c = ConjugacyClass(
                index=class_idx,
                representative=rep,
                members=members,
                size=len(members)
            )
            classes.append(c)
            class_idx += 1
            
        return classes

    def get_class_of(self, element: Hashable) -> ConjugacyClass:
        """Find which class an element belongs to."""
        if element not in self.elem_to_idx:
            raise ValueError("Element not in group")
        
        # Linear search over classes (could be optimised with a cache map)
        for c in self.classes:
            if element in c.members:
                return c
        raise RuntimeError("Element not found in any class (logic error).")


# ============================================================
# 2. Induced Representation Solver
# ============================================================

class InducedRepSolver:
    """
    Manages the subgroup H, the cosets G/H, the induced representation matrices,
    and the interaction graph generation.
    """
    def __init__(self, group: FiniteGroup):
        self.G = group
        self.H_elements = []
        self.cosets = []  # List of frozensets
        self.coset_reps = [] # Representatives
        self.rho_matrices = {} # Map element -> Permutation Matrix
        self.projectors = {}
        self.Qblocks = {}
        self.character_table = None # Dict[class_label -> vector]
        self.irrep_labels = []

    def set_subgroup(self, h_elements: List[Hashable]):
        """Define H by providing a list of its elements."""
        # 1. Verify H is a subset
        for h in h_elements:
            if h not in self.G.elem_to_idx:
                raise ValueError(f"Subgroup element {h} not in G.")
        
        self.H_elements = h_elements
        
        # 2. Compute Left Cosets G/H
        # Logic adapted from original script
        H_indices = {self.G.elem_to_idx[h] for h in h_elements}
        unseen = set(range(self.G.n))
        
        self.cosets = []
        self.coset_reps = []
        
        while unseen:
            r_idx = unseen.pop() # Pick a representative index
            
            # Construct coset: r * H = { r * h for h in H }
            coset_indices = set()
            for h_idx in H_indices:
                val = self.G.mult_table[r_idx, h_idx]
                coset_indices.add(val)
            
            # Store as frozenset of actual elements
            coset_elements = frozenset(self.G.elements[i] for i in coset_indices)
            
            self.cosets.append(coset_elements)
            self.coset_reps.append(self.G.elements[r_idx])
            
            unseen -= coset_indices

        m = len(self.cosets)
        print(f"Computed {m} cosets for H (order {len(h_elements)}) in G (order {self.G.n}).")
        
        # 3. Precompute Permutation Action Matrices rho(g) on G/H
        # We need a map from Coset -> Index to build matrices
        coset_to_idx = {c: i for i, c in enumerate(self.cosets)}
        
        self.rho_matrices = {}
        
        # For every g in G
        for g_idx, g in enumerate(self.G.elements):
            mat = np.zeros((m, m), dtype=float)
            
            # Where does g send coset j?
            # coset j has rep r_j. 
            # g * (r_j H) = (g * r_j) H
            for j, c_set in enumerate(self.cosets):
                # We can pick any element x from c_set to determine the destination
                x = next(iter(c_set))
                gx = self.G.mult_func(g, x)
                
                # Find which coset contains gx
                # Optimisation: In generic code, we scan. 
                # (Can be optimised by pre-mapping every element to its coset index)
                target_idx = -1
                for k, target_c in enumerate(self.cosets):
                    if gx in target_c:
                        target_idx = k
                        break
                
                if target_idx == -1:
                    raise RuntimeError("Coset action failed: destination not found.")
                
                mat[target_idx, j] = 1.0
            
            self.rho_matrices[g] = mat

    def load_character_table(self, mapping: Dict[Hashable, List[complex]], irrep_labels: List[str] = None):
        """
        Load character table via a mapping: { Representative_Element : [chi_0, chi_1, ...] }
        The mapping keys must match at least one member of each computed conjugacy class.
        """
        # Validate that we have a vector for every class
        class_vectors = {}
        num_irreps = 0
        
        for cls in self.G.classes:
            found = False
            vec = None
            
            # Try to find a match in the user provided mapping
            # The user might have provided the exact representative we picked, or another member
            for member in cls.members:
                if member in mapping:
                    vec = np.array(mapping[member], dtype=complex)
                    found = True
                    break
            
            if not found:
                raise ValueError(f"No character vector provided for Class containing {cls.representative}. \n"
                                 f"Class members: {cls.members}")
            
            if num_irreps == 0:
                num_irreps = len(vec)
            elif len(vec) != num_irreps:
                raise ValueError(f"Inconsistent character vector lengths. Expected {num_irreps}, got {len(vec)}.")
                
            class_vectors[cls.index] = vec

        # Store locally
        self.character_table = class_vectors
        self.num_irreps = num_irreps
        if irrep_labels:
            if len(irrep_labels) != num_irreps:
                raise ValueError("Number of labels must match dimension of character vectors.")
            self.irrep_labels = irrep_labels
        else:
            self.irrep_labels = [f"Irrep_{i}" for i in range(num_irreps)]
            
        print("Character table loaded successfully.")

    def compute_projectors(self, tol=1e-10):
        """
        Compute projectors P_alpha = (d_alpha / |G|) * sum_g chi_alpha(g^-1) rho(g)
        """
        if not self.rho_matrices or not self.character_table:
            raise RuntimeError("Subgroup or Character Table not set.")

        m = len(self.cosets) # Dimension of the vector space V = R[G/H]
        group_order = self.G.n
        
        # 1. Pre-sum matrices by class to speed up: S_C = sum_{g in C} rho(g)
        class_sums = {}
        for cls in self.G.classes:
            S = np.zeros((m, m), dtype=complex) # Use complex to be safe
            for g in cls.members:
                S += self.rho_matrices[g]
            class_sums[cls.index] = S

        self.projectors = {}
        self.Qblocks = {}
        
        # For each Irrep alpha (columns in our vectors)
        for alpha_idx in range(self.num_irreps):
            label = self.irrep_labels[alpha_idx]
            
            # Determine dimension d_alpha from the Character of Identity (Class containing identity)
            # Find identity class
            id_class = self.G.get_class_of(self.G.identity)
            chi_vec = self.character_table[id_class.index]
            d_alpha = np.real(chi_vec[alpha_idx]) # Dimension is chi(1)
            
            # Build Matrix M = sum_C chi_alpha(C)* S_C
            # Note: Formula uses chi(g^-1). 
            # For finite groups, chi(g^-1) is complex conjugate of chi(g).
            M = np.zeros((m, m), dtype=complex)
            
            for cls in self.G.classes:
                # Get char value for this class and this irrep
                val = self.character_table[cls.index][alpha_idx]
                val_conj = np.conj(val) # chi(g^-1)
                
                M += val_conj * class_sums[cls.index]
            
            # P = (d / |G|) * M
            P = (d_alpha / group_order) * M
            
            # If characters are real, P should be real. Enforce if close.
            if np.allclose(np.imag(P), 0, atol=tol):
                P = np.real(P)
                # Symmetrise to fix numerical noise
                P = 0.5 * (P + P.T)
            
            self.projectors[label] = P
            
            # Eigendecomposition to find subspace
            # We look for eigenvalues near 1
            vals, vecs = np.linalg.eigh(P) # eigh for Hermitian/Symmetric
            idx_ones = np.where(np.abs(vals - 1.0) < tol)[0]
            
            if len(idx_ones) > 0:
                self.Qblocks[label] = vecs[:, idx_ones]
            else:
                self.Qblocks[label] = np.array([]) # Empty array if irrep not present

    def build_interaction_graph(self, activation_fn, tol=1e-8, verbose=False) -> nx.DiGraph:
        """
        Build the interaction graph between Irreps present in the induced representation.
        """
        # Filter irreps that actually exist in the decomposition (non-empty Qblocks)
        present_labels = [L for L, Q in self.Qblocks.items() if Q.size > 0 and Q.ndim > 1]
        
        if verbose:
            dims = {L: self.Qblocks[L].shape[1] for L in present_labels}
            print("Irreps present in G/H:", dims)
            
        G_graph = nx.DiGraph()
        G_graph.add_nodes_from(present_labels)
        
        phi = np.vectorize(activation_fn)
        
        def test_edge(L_src, L_dst):
            Q_src = self.Qblocks[L_src]
            Q_dst = self.Qblocks[L_dst]
            
            # Logic: Check if phi(v_src) has component in subspace_dst
            # Check basis vectors of src
            for a in range(Q_src.shape[1]):
                v = Q_src[:, a]
                v_act = phi(v)
                # Project onto dst: coeffs = Q_dst.T @ v_act
                coeffs = Q_dst.T @ v_act
                if np.any(np.abs(coeffs) > tol):
                    return True
            
            # Check sums of pairs (captures non-linearity better)
            for a in range(Q_src.shape[1]):
                for b in range(a+1, Q_src.shape[1]):
                    v = Q_src[:, a] + Q_src[:, b]
                    v_act = phi(v)
                    coeffs = Q_dst.T @ v_act
                    if np.any(np.abs(coeffs) > tol):
                        return True
            return False

        for src in present_labels:
            for dst in present_labels:
                if test_edge(src, dst):
                    G_graph.add_edge(src, dst)
                    
        return G_graph


# ============================================================
# 3. Main / Demo Logic (Replicating A5 case)
# ============================================================

def visualise_graph(G, title="Interaction Graph"):
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, pos, node_color="#E8F0FF", edgecolors="blue", 
                     node_size=1000, font_weight="bold", with_labels=True)
    plt.title(title)
    plt.axis("off")
    plt.show()
