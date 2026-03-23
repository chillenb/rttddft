import scipy
import numpy as np
from pyscf import lib

nocc = 4
nvir = 5
naux = 10

Pia = np.random.random((naux, nocc, nvir)) + 1j * np.random.random((naux, nocc, nvir))
Pi = np.zeros((naux, naux), dtype=np.complex128)

scipy.linalg.blas.zherk(
    alpha=1.0,
    a=Pia.reshape(naux, nocc * nvir).T,
    c=Pi.T,
    trans=2,
    beta=1.0,
    overwrite_c=True,
)

# Since trans=2, C=a^H a
#                 = Pia.reshape(naux, nocc * nvir).T.conj() @ Pia.reshape(naux, nocc * nvir)
#                 = einsum('Qia, Pia->QP', Pia.conj(), Pia)
# but C = Pi.T
# therefore Pi = np.einsum('Pia, Qia->PQ', Pia, Pia.conj())

lib.hermi_triu(Pi, inplace=True)

err = np.linalg.norm(Pi - np.einsum('Pia, Qia->PQ', Pia, Pia.conj()))
assert err < 1e-6