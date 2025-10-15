import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sympy as sp
from scipy.special import erf
import math
import networkx as nx


def is_zero_numeric(A, tol=1e-4):
    """Return True if the numeric matrix/vector is zero up to tol."""
    return np.linalg.norm(np.asarray(A, dtype=float)) <= tol

def chop_matrix(B, tol=1e-4):
    """Return a SymPy matrix with tiny entries set to 0."""
    B = sp.Matrix(B)
    return B.applyfunc(lambda x: 0 if abs(float(x)) < tol else x)

## Calculation Code

def q_matrix(n: int, normalise: bool = False, j_start: int = 1):
    '''
    Change of basis matrix to R^n
    '''
        
    j = sp.symbols('j', integer=True)
    k = sp.symbols('k', integer=True)
    pi = sp.pi
    
    cols = []
    labels = [] 
    
    def col(vec):
        return sp.Matrix(vec)
    
    # j-range
    js = list(range(j_start, j_start + n))
    
    # w_0: trivial rep
    w0 = [sp.Integer(1) for _ in js]
    cols.append(col(w0))
    labels.append(("L_0", "w_0"))
    
    # 2D rotation pairs
    max_k = (n - 1) // 2 
    for kk in range(1, max_k + 1):
        w_plus = [ -sp.sin(2*pi*jj*kk/n) for jj in js ]  # (-sin)
        w_minus = [  sp.cos(2*pi*jj*kk/n) for jj in js ]  # (cos)
        cols.append(col(w_plus))
        labels.append((f"L_{kk}", f"w_plus_{kk}"))
        cols.append(col(w_minus))
        labels.append((f"L_{kk}", f"w_minus_{kk}"))
    
    # w_m: sign rep if n even: k = n/2
    if n % 2 == 0:
        m = n // 2
        w_m = [ sp.cos(sp.pi*jj) for jj in js ]  # (-1)^j
        cols.append(col(w_m))
        labels.append((f"L_{m}", f"w_{m}"))
    

    Q = sp.Matrix.hstack(*cols)
    
    # Normalisation of columns (optional)
    if normalise:
        Qn_cols = []
        new_labels = []
        for c, lab in zip(Q.T, labels):
            norm = sp.sqrt(sp.nsimplify(sum(ci**2 for ci in c)))
            norm = sp.simplify(norm)

            if norm != 0:
                Qn_cols.append(c / norm)
            else:
                Qn_cols.append(c)
            new_labels.append(lab)
        Q = sp.Matrix.hstack(*Qn_cols)
        labels = new_labels
    
    return Q, labels


def m_matrix(n: int):
    """
    Regular rep of C_n in permutation basis.
    """

    M = sp.zeros(n, n)
    for j in range(n): 
        i = (j + 1) % n
        M[i, j] = 1

    return M


def d_matrix(n:int):
    """Diagonal D so that Q^{-1} = D Q^T"""
    entries = [sp.Rational(1, n)]  # trivial column
    for _ in range((n - 1)//2):    # each 2D pair
        entries += [sp.Rational(2, n), sp.Rational(2, n)]
    if n % 2 == 0:                 # sign column
        entries.append(sp.Rational(1, n))
    return sp.diag(*entries)


def indices_by_rep(labels):
    """
    Group Q's columns by simple L.
    labels is the second return of q_matrix(n, ...).
    Returns: dict like {'L_0':[0], 'L_1':[1,2], 'L_2':[3,4], 'L_sign':[...]}
    """
    idx = {}
    for j,(L,_) in enumerate(labels):
        idx.setdefault(L, []).append(j)
    return idx



def inclusion_matrix(Q, cols):
    """i_L: R^{dim L} -> R^n (select the columns for L)."""
    return Q[:, cols]



def projection_matrix(Q, D, cols):
    """p_L: R^n -> R^{dim L} (select the matching rows of Q^{-1} = D Q^T)."""
    Q_inv = D * Q.T
    return Q_inv[cols, :]



def activation_func_block(Q, D, byL, src_L, dst_L, activation_func,  test_vectors=None):
    """
    Compute the block map p o f o i by evaluating on basis/test vectors.
    Returns a matrix of shape (dim dst_L, dim src_L) if using the canonical basis of L,
    or a list of column-vectors if you pass custom test_vectors.
    """
    iL = inclusion_matrix(Q, byL[src_L])    # (n, d_src)
    pK = projection_matrix(Q, D, byL[dst_L])# (d_dst, n)

    d_src = iL.shape[1]

    if test_vectors is None:
        test_vectors = [sp.Matrix.eye(d_src)[:, j] for j in range(d_src)] # standard basis of R^n

    cols = []
    for v in test_vectors:
        x = iL * v                           # embed in R^n
        x_np = np.array(x, dtype=float).reshape(-1)
        y_np = activation_func(x_np)                    # apply activation function coordinatewise
        y = sp.Matrix(y_np)                  # back to sympy
        out = pK * y                         # project to dst_L
        out = chop_matrix(out, tol=1e-12)
        cols.append(out)

    if test_vectors is None or all(vec.shape == (d_src,1) for vec in test_vectors):
        B = sp.Matrix.hstack(*cols)          # (d_dst, d_src)
        B = chop_matrix(B, tol=1e-12)
        return B
    return cols

