import itertools
from collections import deque
import sympy as sp


# ============================================================
# Basic group builders in the same style as the example code
# ============================================================

def build_Cn(n):
    elements = list(range(n))

    def mult(a, b):
        return (a + b) % n

    return elements, mult


def build_D2n(n):
    """
    Encode D_{2n} as pairs (a,b) with a in Z/nZ and b in {0,1},
    interpreted as r^a s^b.

    Multiplication:
        (a,b)(c,d) = (a + (-1)^b c, b+d mod 2)
    corresponding to the relation s r s = r^{-1}.
    """
    elements = [(a, b) for a in range(n) for b in (0, 1)]

    def mult(x, y):
        a, b = x
        c, d = y
        return ((a + ((-1) ** b) * c) % n, (b + d) % 2)

    return elements, mult


def build_Sn(n):
    elements = list(itertools.permutations(range(n)))

    def mult(a, b):
        # Same convention as your example file.
        return tuple(a[i] for i in b)

    return elements, mult


# ============================================================
# 1. C_n irreps
# ============================================================

def irreps_mats_Cn(n, labels=None):
    """
    Returns
        irrep_mats[label][g] = SymPy matrix
    with labels matching the example C_n character table labels:
        \chi_0, \chi_1, ..., \chi_{n-1}
    """
    zeta = sp.exp(2 * sp.pi * sp.I / n)
    out = {}

    for j in range(n):
        if labels is not None:
            label = labels[j]
        else:
            label = str(j)
        rho = {}
        for a in range(n):
            rho[a] = sp.Matrix([[sp.simplify(zeta ** (j * a))]])
        out[label] = rho

    return out


# ============================================================
# 2. D_{2n} irreps
# ============================================================

def irreps_mats_D2n(n):
    """
    Uses the encoding (a,b) = r^a s^b from build_D2n().

    Labels used here:
      - odd n:
            chi_r+1_s+1, chi_r+1_s-1,
            rho_1, ..., rho_{(n-1)//2}
      - even n:
            chi_r+1_s+1, chi_r+1_s-1,
            chi_r-1_s+1, chi_r-1_s-1,
            rho_1, ..., rho_{n//2 - 1}

    Make sure your D_{2n} character-table labels match these exactly.
    """
    elements, _ = build_D2n(n)
    out = {}

    # 1-dimensional irreps
    possible_r_vals = [1, -1] if n % 2 == 0 else [1]
    possible_s_vals = [1, -1]

    for eps_r in possible_r_vals:
        for eps_s in possible_s_vals:
            label = rf"$\chi_{{r={eps_r},s={eps_s}}}$"
            rho = {}
            for a, b in elements:
                rho[(a, b)] = sp.Matrix([[sp.Integer(eps_r) ** a * sp.Integer(eps_s) ** b]])
            out[label] = rho

    # 2-dimensional irreps
    max_k = (n // 2 - 1) if n % 2 == 0 else ((n - 1) // 2)
    S = sp.Matrix([[1, 0], [0, -1]])
    I2 = sp.eye(2)

    for k in range(1, max_k + 1):
        theta = 2 * sp.pi * k / n
        rho = {}
        for a, b in elements:
            R = sp.Matrix([
                [sp.cos(a * theta), -sp.sin(a * theta)],
                [sp.sin(a * theta),  sp.cos(a * theta)],
            ])
            rho[(a, b)] = sp.simplify(R * (S if b else I2))
        out[rf"$\rho_{{{k}}}$"] = rho

    return out


# ============================================================
# 3. S_n irreps via Young's seminormal / orthogonal form
# ============================================================

def generate_partitions(n):
    def rec(rem, max_part, current, result):
        if rem == 0:
            result.append(tuple(current))
            return
        for i in range(min(max_part, rem), 0, -1):
            rec(rem - i, i, current + [i], result)

    result = []
    rec(n, n, [], result)
    return result


def partition_labels(n):
    return [''.join(map(str, p)) for p in generate_partitions(n)]


def _tableau_from_positions(shape, pos):
    rows = [[None] * shape[r] for r in range(len(shape))]
    for num, (r, c) in pos.items():
        rows[r][c] = num
    return tuple(tuple(row) for row in rows)


def _column_reading(T):
    max_c = max(len(row) for row in T)
    out = []
    for c in range(max_c):
        for r in range(len(T)):
            if c < len(T[r]):
                out.append(T[r][c])
    return tuple(out)



def standard_tableaux(shape):
    """
    Returns all standard tableaux of shape 'shape', sorted by column-reading.
    This matches the ordering convention described in the symmetric-group addendum.
    """
    cells = [(r, c) for r, row_len in enumerate(shape) for c in range(row_len)]
    n = sum(shape)
    used = set()
    pos = {}
    out = []

    def backtrack(num):
        if num == n + 1:
            out.append(_tableau_from_positions(shape, pos.copy()))
            return

        for r, c in cells:
            if (r, c) in used:
                continue
            # To keep rows increasing, the left neighbour must already be filled.
            if c > 0 and (r, c - 1) not in used:
                continue
            # To keep columns increasing, the upper neighbour must already be filled.
            if r > 0 and c < shape[r - 1] and (r - 1, c) not in used:
                continue

            used.add((r, c))
            pos[num] = (r, c)
            backtrack(num + 1)
            used.remove((r, c))
            del pos[num]

    backtrack(1)
    out.sort(key=_column_reading)
    return out



def _positions(T):
    out = {}
    for r, row in enumerate(T):
        for c, val in enumerate(row):
            out[val] = (r, c)
    return out



def _swap_entries(T, a, b):
    rows = [list(row) for row in T]
    pa = pb = None
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            if val == a:
                pa = (r, c)
            elif val == b:
                pb = (r, c)
    ra, ca = pa
    rb, cb = pb
    rows[ra][ca], rows[rb][cb] = rows[rb][cb], rows[ra][ca]
    return tuple(tuple(row) for row in rows)



def _is_standard(T):
    # Rows
    for row in T:
        for j in range(len(row) - 1):
            if not row[j] < row[j + 1]:
                return False
    # Columns
    max_c = max(len(row) for row in T)
    for c in range(max_c):
        col = []
        for r in range(len(T)):
            if c < len(T[r]):
                col.append(T[r][c])
        for j in range(len(col) - 1):
            if not col[j] < col[j + 1]:
                return False
    return True



def seminormal_generator_matrix(shape, tableaux, i):
    """
    Matrix of the simple transposition s_i = (i, i+1)
    in Young's orthogonal form, in the basis of standard tableaux.

    Here i is 1-based: i = 1, ..., n-1.
    """
    d = len(tableaux)
    idx = {T: j for j, T in enumerate(tableaux)}
    M = sp.zeros(d)

    for col_idx, T in enumerate(tableaux):
        pos = _positions(T)
        ri, ci = pos[i]
        rj, cj = pos[i + 1]

        # same row => +1
        if ri == rj:
            M[col_idx, col_idx] = 1
            continue

        # same column => -1
        if ci == cj:
            M[col_idx, col_idx] = -1
            continue

        # generic 2x2 seminormal block
        axial = (cj - rj) - (ci - ri)
        coeff = sp.Rational(1, axial)
        off = sp.sqrt(1 - coeff ** 2)

        T_swapped = _swap_entries(T, i, i + 1)
        if not _is_standard(T_swapped):
            raise ValueError("Swapped tableau should be standard in the generic case.")

        row_idx = idx[T_swapped]
        M[col_idx, col_idx] = coeff
        M[row_idx, col_idx] = off

    return sp.simplify(M)



def irreps_mats_Sn(n):
    """
    Returns irreducible matrices for S_n labelled by partitions,
    with labels matching your example character-table labels:
        'n', 'n-1,1' rendered as '31', '211', etc.

    This constructs one Specht-module realisation for each partition using
    Young's seminormal / orthogonal form.
    """
    elements, mult = build_Sn(n)
    parts = generate_partitions(n)
    labels = [''.join(map(str, p)) for p in parts]

    # adjacent transpositions in the same encoding as build_Sn()
    simple_generators = []
    for i in range(n - 1):
        s = list(range(n))
        s[i], s[i + 1] = s[i + 1], s[i]
        simple_generators.append(tuple(s))

    out = {}

    for shape, label in zip(parts, labels):
        tableaux = standard_tableaux(shape)
        d = len(tableaux)
        identity = tuple(range(n))

        gen_mats = {}
        for i, s in enumerate(simple_generators, start=1):
            gen_mats[s] = seminormal_generator_matrix(shape, tableaux, i)

        # Build rho(g) for all g by traversing the Cayley graph.
        rho = {identity: sp.eye(d)}
        q = deque([identity])

        while q:
            g = q.popleft()
            for s in simple_generators:
                h = mult(g, s)
                if h not in rho:
                    # rho[h] = sp.simplify(rho[g] * gen_mats[s])
                    rho[h] = rho[g] * gen_mats[s]
                    q.append(h)

        out[label] = rho

    return out


# ============================================================
# Convenience wrapper
# ============================================================

def generate_irrep_mats(group_type, n):
    if group_type == "Cn":
        return irreps_mats_Cn(n)
    if group_type == "D2n":
        return irreps_mats_D2n(n)
    if group_type == "Sn":
        return irreps_mats_Sn(n)
    raise ValueError("group_type must be one of: 'Cn', 'D2n', 'Sn'")
