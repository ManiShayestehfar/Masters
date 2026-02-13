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




