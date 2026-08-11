"""Verify that Pi_j X_j commutes exactly with H_neuron (D-9).

Run: python3 src/symmetry_check.py
Requires: numpy only.
"""
import numpy as np, itertools

def build(L, N, D, J, G):
    a  = np.diag(np.sqrt(np.arange(1,N)),1)           # annihilation, truncated
    x  = a + a.conj().T                               # a + a^dag
    P  = np.zeros((N,N)); P[N-1,N-1] = 1.0            # |N-1><N-1|
    IN = np.eye(N); I2 = np.eye(2)
    X  = np.array([[0,1],[1,0]],float)
    Y  = np.array([[0,-1j],[1j,0]])
    Z  = np.array([[1,0],[0,-1]],float)
    sp = np.array([[0,1],[0,0]],float); sm = sp.T     # sigma+ , sigma-

    d = (2*N)**L
    def emb(ops):                                      # ops: dict site -> (bos_op, spin_op)
        M = np.array([[1.0+0j]])
        for j in range(L):
            b,s = ops.get(j,(IN,I2))
            M = np.kron(M, np.kron(b,s))
            
        return M

    H = np.zeros((d,d),complex)
    for j in range(L):
        H += D[j]*emb({j:(x,I2)})                                        # H_boson
        H += G*emb({j:(P,X)})                                            # H_CNOT
    for j in range(L-1):
        H += J*(emb({j:(IN,sp), j+1:(IN,sm)}) + emb({j:(IN,sm), j+1:(IN,sp)}))   # H_spin

    PiX = emb({j:(IN,X) for j in range(L)})
    PiZ = emb({j:(IN,Z) for j in range(L)})
    return H, PiX, PiZ

for (L,N) in [(2,4),(3,3),(3,5),(4,3)]:
    rng = np.random.default_rng(0)
    D = rng.uniform(0.3,1.2,L)                        # site-dependent D_j
    H,PiX,PiZ = build(L,N,D,J=0.63,G=0.91)
    cX = np.linalg.norm(H@PiX - PiX@H)
    cZ = np.linalg.norm(H@PiZ - PiZ@H)
    print(f"L={L} N={N} dim={H.shape[0]:5d} ||[H,PiX]||={cX:.3e}   ||[H,PiZ]||={cZ:.3e}")

# is the natural initial state (all bosons |0>, all spins down) in a PiX sector?
L,N = 3,4
H,PiX,PiZ = build(L,N,np.array([0.5,0.8,1.1]),0.63,0.91)
v = np.zeros((2*N)**L, complex)
idx = 0
for j in range(L): idx = idx*(2*N) + 0                # boson n=0, spin |0> -> local index 0
v[idx] = 1.0
w = PiX@v
print("\nnatural init state |0..0>|down..down>:")
print("  overlap with PiX|psi> =", np.round(np.vdot(v,w).real,6), " (=+-1 iff eigenstate)")
ev = np.vdot(v, PiX@v).real
print("  <PiX> =", round(ev,6), "-> equal superposition of both parity sectors" if abs(ev)<1e-9 else "")
