"""Check the JW convention of Appendix A term by term.

Appendix A states:  a_j = (prod_{k<j} sx_k) sz_j ,  b_j = (prod_{k<j} sx_k) sy_j
and claims          H = - sum_alpha t_alpha sum_j  i b_j a_{j+alpha}
reproduces          H = - t0 sum sx - t1 sum sz sz - sum_{a>=2} t_a sum sz sx..sx sz
"""
import numpy as np, itertools
I=np.eye(2); X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1,-1]).astype(complex)
def kron(ops):
    M=np.array([[1]],complex)
    for o in ops: M=np.kron(M,o)
    return M
L=6
def op(site,P):        # single-site Pauli
    return kron([P if i==site else I for i in range(L)])
def a_maj(j):          # a_j = (prod_{k<j} X_k) Z_j
    return kron([X if k<j else (Z if k==j else I) for k in range(L)])
def b_maj(j):          # b_j = (prod_{k<j} X_k) Y_j
    return kron([X if k<j else (Y if k==j else I) for k in range(L)])
def spin_term(alpha,j):
    """the alpha-term of Eq.(9) on site j, WITHOUT the -t_alpha prefactor"""
    if alpha==0: return op(j,X)
    if alpha==1: return op(j,Z)@op(j+1,Z)
    ops=[I]*L; ops[j]=Z; ops[j+alpha]=Z
    for m in range(j+1,j+alpha): ops[m]=X
    return kron(ops)
print("  alpha |  i b_j a_{j+alpha}  vs  spin term   ->  ratio")
for alpha in range(0,4):
    j=1
    maj = 1j*b_maj(j)@a_maj(j+alpha)
    spin = spin_term(alpha,j)
    nz = np.abs(spin)>1e-9
    r = maj[nz]/spin[nz]
    r = r[0] if np.allclose(r, r[0]) else "not proportional"
    print(f"    {alpha}   |  ratio = {r}")
