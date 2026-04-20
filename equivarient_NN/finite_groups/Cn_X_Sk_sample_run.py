# ============================================================
# ENTRY SCRIPT FOR:
#   - build G = C_n x S_k
#   - compute induced permutation rep on G/H
#   - decompose via characters
#   - (optional) refine irreps via matrices
#   - build isotypic / interaction graph
# ============================================================

# ---- imports ------------------------------------------------
# Core deps: sympy (symbolic), numpy (numeric), networkx (graph)
# Local modules:
#   FiniteGroup → group structure + conjugacy classes
#   InducedRepSolver → main decomposition + graph builder
#   irreps_mats_* → explicit irreps (needed ONLY if refine=True)

import sys
sys.path.append('../')

import itertools
import sympy as sp
import numpy as np

from finite_groups import FiniteGroup
from induced_rep_solver import InducedRepSolver
from symchar.symchar import character_table

from groups.irrep_mats_generators import irreps_mats_Cn, irreps_mats_Sn


# ============================================================
# ACTIVATIONS (used in graph construction)
# ============================================================

# Must accept sympy scalar → return sympy expression
# Later lambdified to numpy inside solver

def RELU(x): return sp.Max(0, x)
def TANH(x): return sp.tanh(x)
def SIGMOID(x): return 1 / (1 + sp.exp(-x))
def LINEAR(x): return x
def SQUARE(x): return x**2
def RELU_TANH(x): return RELU(x) + TANH(x)


# ============================================================
# GROUP BUILDERS
# ============================================================

# All groups are wrapped in FiniteGroup:
#   - elements: list
#   - mult_func: closure
#   - auto builds Cayley table + conjugacy classes

def build_Cn(n):
    # cyclic group Z/nZ
    elements = list(range(n))
    def mult(a, b): return (a + b) % n
    return FiniteGroup(elements, mult)


def build_Sk(k):
    # symmetric group on k letters
    # permutations encoded as tuples
    elements = list(itertools.permutations(range(k)))
    def mult(a, b): return tuple(a[i] for i in b)
    return FiniteGroup(elements, mult)


def build_product(G1, G2):
    # direct product G1 x G2
    # elements: (g, h)
    elements = [(g, h) for g in G1.elements for h in G2.elements]

    def mult(x, y):
        g1, h1 = x
        g2, h2 = y
        return (G1.mult_func(g1, g2), G2.mult_func(h1, h2))

    return FiniteGroup(elements, mult)


def build_H(k):
    """
    Subgroup H ⊂ C_n × S_k used for induction.
    Here:
        H ≅ S_{k-1} embedded as permutations fixing last index.
        First component fixed to 0 in C_n.
    """
    perms = list(itertools.permutations(range(k - 1)))
    H = []

    for p in perms:
        perm = list(range(k))
        for i in range(k - 1):
            perm[i] = p[i]
        H.append((0, tuple(perm)))

    return H


# ============================================================
# PARTITIONS / CYCLE TYPES (S_k structure)
# ============================================================

def generate_partitions(n):
    # all integer partitions of n (descending)
    def generate(n, max_part, current, result):
        if n == 0:
            result.append(tuple(current))
            return
        for i in range(min(max_part, n), 0, -1):
            generate(n - i, i, current + [i], result)

    result = []
    generate(n, n, [], result)
    return result


def cycle_type(perm):
    # convert permutation → partition of cycle lengths
    k = len(perm)
    seen = [False] * k
    parts = []

    for i in range(k):
        if not seen[i]:
            j = i
            length = 0
            while not seen[j]:
                seen[j] = True
                j = perm[j]
                length += 1
            parts.append(length)

    return tuple(sorted(parts, reverse=True))


# ============================================================
# CHARACTER TABLES
# ============================================================

def Sk_character_table_map(G: FiniteGroup, k: int):
    """
    S_k characters (via symchar)

    OUTPUT:
        class_char_map[rep] = row
        labels = partitions
        partition_map = partition → row index

    IMPORTANT:
        Keys MUST be cls.representative from SAME group G
        (required by solver.load_character_table)
    """
    raw = np.array(character_table(k), dtype=object).T

    partitions = generate_partitions(k)
    labels = [''.join(map(str, p)) for p in partitions]

    partition_map = {
        tuple(sorted(p, reverse=True)): i
        for i, p in enumerate(partitions)
    }

    class_char_map = {}

    for cls in G.classes:
        rep = cls.representative
        part = cycle_type(rep)

        if part not in partition_map:
            raise ValueError(f"Cycle type {part} not found")

        class_char_map[rep] = [
            sp.sympify(x) for x in raw[partition_map[part]]
        ]

    return class_char_map, labels, partition_map


def Cn_character_table(n):
    """
    Characters of C_n:
        χ_j(g) = exp(2πi jg/n)

    LABEL FORMAT MUST MATCH irreps_mats_Cn:
        "$\\chi_0$", "$\\chi_1$", ...
    """
    table = []

    for g in range(n):
        row = [sp.exp(2 * sp.pi * sp.I * g * m / n) for m in range(n)]
        table.append(row)

    labels = [rf"\chi_{j}" for j in range(n)]

    return np.array(table, dtype=object), labels


# ============================================================
# DIRECT PRODUCT CHARACTER TABLE
# ============================================================

def direct_prod_character_table(G,
                               Cn_table, Cn_labels,
                               Sk_table, Sk_labels,
                               Sk_partition_map=None):
    """
    Build character table for C_n × S_k.

    Supports:
        - Sk_table as dict (preferred)
        - Sk_table as matrix (legacy)

    Output:
        class_char_map[(g,σ)] = tensor-product row
        labels = χ ⊗ ψ
    """
    labels = [rf"${c} \otimes {s}$" for c in Cn_labels for s in Sk_labels]
    class_char_dict = {}

    sk_is_map = isinstance(Sk_table, dict)

    for cls in G.classes:
        g, sigma = cls.representative

        cn_row = Cn_table[g]

        if sk_is_map:
            sk_row = Sk_table[sigma]
        else:
            part = cycle_type(sigma)
            sk_row = Sk_table[Sk_partition_map[part]]

        # tensor product of characters
        row = [sp.simplify(chi * psi) for chi in cn_row for psi in sk_row]

        class_char_dict[cls.representative] = row

    return class_char_dict, labels


# ============================================================
# PRODUCT IRREPS (REQUIRED FOR refine=True)
# ============================================================

def direct_prod_irrep_mats(Cn_n, Sk_n, Cn_labels, Sk_labels):
    """
    Build irreps of C_n × S_k from tensor product:

        ρ(g,σ) = χ(g) · ρ_Sk(σ)

    REQUIREMENT:
        labels MUST EXACTLY match direct_prod_character_table labels
        (solver indexes by string equality)
    """
    Cn_irreps = irreps_mats_Cn(Cn_n)
    Sk_irreps = irreps_mats_Sn(Sk_n)

    out = {}

    for c in Cn_labels:
        for s in Sk_labels:

            label = rf"${c} \otimes {s}$"

            rho_c = Cn_irreps[c]   # scalar
            rho_s = Sk_irreps[s]   # matrix

            rho_prod = {}

            for g in rho_c:
                for sigma in rho_s:
                    rho_prod[(g, sigma)] = sp.simplify(
                        rho_c[g][0, 0] * rho_s[sigma]
                    )

            out[label] = rho_prod

    return out


# ============================================================
# MAIN PIPELINE
# ============================================================

def run(Cn, Sk, H, n, activation_fn,
        refine=False,
        regular=False,
        irrep_mats=None,
        interaction=False,
        figsize=(10, 10),
        show_kernel_inclusions=True,
        show_self_loops=False):
    """
    Pipeline:

    1. Build G = C_n × S_k
    2. Build permutation rep on G/H
    3. Load character table
    4. (optional) load irreps for refinement
    5. Compute projectors
    6. Build graph (isotypic or interaction)
    """

    G = build_product(Cn, Sk)

    solver = InducedRepSolver(G)

    # subgroup choice
    if regular:
        solver.set_subgroup([G.identity])   # regular representation
    else:
        solver.set_subgroup(H)              # induced representation

    # character tables
    Cn_table, Cn_labels = Cn_character_table(n[0])
    Sk_char_map, Sk_labels, Sk_partition_map = Sk_character_table_map(Sk, n[1])

    class_char_map, labels = direct_prod_character_table(
        G,
        Cn_table, Cn_labels,
        Sk_char_map, Sk_labels,
        Sk_partition_map
    )

    solver.load_character_table(class_char_map, labels)

    # refinement requires explicit irreps
    if refine:
        if irrep_mats is None:
            irrep_mats = direct_prod_irrep_mats(n[0], n[1], Cn_labels, Sk_labels)
        solver.load_irrep_matrices(irrep_mats)

    solver.compute_projectors(refine=refine)

    # graph type
    if interaction:
        graph = solver.build_interaction_graph(activation_fn=activation_fn)
    else:
        graph = solver.build_isotypic_graph(activation_fn=activation_fn)

    solver.visualise_interaction_graph(
        graph,
        group_name=rf"$C_{n[0]} \times S_{n[1]}$",
        node_size=2200,
        show_self_loops=show_self_loops,
        show_kernel_inclusions=show_kernel_inclusions,
        figsize=figsize
    )

    return graph, solver


# ============================================================
# EXAMPLE RUN
# ============================================================

if __name__ == "__main__":
    n = [4, 4]

    Cn = build_Cn(n[0])
    Sk = build_Sk(n[1])
    H = build_H(n[1])

    graph, solver = run(
        Cn, Sk, H, n,
        activation_fn=RELU,
        regular=False,
        refine=False
    )