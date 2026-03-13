import numpy as np
from functools import reduce

def chaindot(*args):
    return reduce(np.matmul, args)


import mpi4py.MPI as MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from pyscf.pbc.mpitools.mpi_helper import allreduce_inplace_contiguous

def my_kpts_and_inds(kpts):
    my_kpt_inds = np.arange(rank, len(kpts), size)
    return kpts[my_kpt_inds], my_kpt_inds


class MPIKBasisChanger:
    """Basis change helper class.
    
    Example:
        >>> from pyscf import gto, scf
        >>> import numpy as np
        >>> from rttddft.lib import BasisChanger
        >>> mol = gto.M(atom="C 0 0 0; O 0 0 1.128", basis='ccpvdz', verbose=5)
        >>> mf = scf.RHF(mol)
        >>> mf.kernel()
        >>> C_mo_ao = mf.mo_coeff
        >>> fock = mf.get_fock(dm=mydensity)
        >>> S_ao = mol.intor('int1e_ovlp')
        >>> ao2mo = BasisChanger(S_ao, C_mo_ao)
        >>> fock_mo = ao2mo.rotate_focklike(fock)        
    """
    def __init__(self, S, C, to_orthonormal=False, nkpts=None):
        """
        Construct an instance of the basis change helper class.

        Args:
            S (np.ndarray): overlap matrix in original basis
            C (np.ndarray): transformation matrix.
            
            The columns of C are the new basis vectors expressed in the original basis.
            
        """
        if nkpts is None:
            if S.ndim == 3:
                nkpts = S.shape[0]
            else:
                raise ValueError('k-point basis changer requires S.ndim == 3')
        self.nkpts = nkpts
        self.S = S
        self.C = C
        self.to_orthonormal = to_orthonormal

        self.Cinv = np.zeros_like(C)
        _, my_kpt_inds = my_kpts_and_inds(np.arange(nkpts))
        for k in my_kpt_inds:
            if to_orthonormal:
                self.Cinv[k] = np.dot(C[k].conj().T, S[k])
            else:
                self.Cinv[k] = np.linalg.inv(C[k])
        
        allreduce_inplace_contiguous(self.Cinv)
        
    def rotate_focklike(self, kmat):
        kmat_transformed = np.zeros_like(kmat)
        _, my_kpt_inds = my_kpts_and_inds(np.arange(self.nkpts))
        for k in my_kpt_inds:
            kmat_transformed[k] = chaindot(self.C[k].conj().T, kmat[k], self.C[k])
        allreduce_inplace_contiguous(kmat_transformed)
        return kmat_transformed

    def rotate_denslike(self, kmat):
        kmat_transformed = np.zeros_like(kmat)
        _, my_kpt_inds = my_kpts_and_inds(np.arange(self.nkpts))
        for k in my_kpt_inds:
            kmat_transformed[k] = chaindot(self.Cinv[k], kmat[k], self.Cinv[k].conj().T)
        allreduce_inplace_contiguous(kmat_transformed)
        return kmat_transformed


    def rotate_oplike(self, kmat):
        kmat_transformed = np.zeros_like(kmat)
        _, my_kpt_inds = my_kpts_and_inds(np.arange(self.nkpts))
        for k in my_kpt_inds:
            kmat_transformed[k] = chaindot(self.Cinv[k], kmat[k], self.C[k])
        allreduce_inplace_contiguous(kmat_transformed)
        return kmat_transformed
    
    def rev_focklike(self, kmat):
        kmat_transformed = np.zeros_like(kmat)
        _, my_kpt_inds = my_kpts_and_inds(np.arange(self.nkpts))
        for k in my_kpt_inds:
            kmat_transformed[k] = chaindot(self.Cinv[k].conj().T, kmat[k], self.Cinv[k])
        allreduce_inplace_contiguous(kmat_transformed)
        return kmat_transformed

    def rev_denslike(self, kmat):
        kmat_transformed = np.zeros_like(kmat)
        _, my_kpt_inds = my_kpts_and_inds(np.arange(self.nkpts))
        for k in my_kpt_inds:
            kmat_transformed[k] = chaindot(self.C[k], kmat[k], self.C[k].conj().T)
        allreduce_inplace_contiguous(kmat_transformed)
        return kmat_transformed

    def rev_oplike(self, kmat):
        kmat_transformed = np.zeros_like(kmat)
        _, my_kpt_inds = my_kpts_and_inds(np.arange(self.nkpts))
        for k in my_kpt_inds:
            kmat_transformed[k] = chaindot(self.C[k], kmat[k], self.Cinv[k])
        allreduce_inplace_contiguous(kmat_transformed)
        return kmat_transformed

    def transform(self, kmat, mat_type='focklike', rev=False):
        if not rev:
            if mat_type == 'focklike':
                return self.rotate_focklike(kmat)
            elif mat_type == 'denslike':
                return self.rotate_denslike(kmat)
            elif mat_type == 'oplike':
                return self.rotate_oplike(kmat)
            else:
                raise ValueError(f'Unknown mat_type {mat_type}')
        else:
            if mat_type == 'focklike':
                return self.rev_focklike(kmat)
            elif mat_type == 'denslike':
                return self.rev_denslike(kmat)
            elif mat_type == 'oplike':
                return self.rev_oplike(kmat)
            else:
                raise ValueError(f'Unknown mat_type {mat_type}')
        
    
    def inverse(self):
        if self.to_orthonormal:
            S_tilde = np.zeros_like(self.S)
            _, my_kpt_inds = my_kpts_and_inds(np.arange(self.nkpts))
            for k in my_kpt_inds:
                S_tilde[k] = np.eye(self.S.shape[1])
            allreduce_inplace_contiguous(S_tilde)
        else:
            S_tilde = self.rotate_focklike(self.S)

        if np.all(np.linalg.norm(self.S - np.eye(self.S.shape[0]), axis=0) < 1.0e-8):
            return MPIKBasisChanger(S_tilde, self.Cinv, to_orthonormal=True)
        return MPIKBasisChanger(S_tilde, self.Cinv)
    
    def chain(self, other):
        S = self.S
        C = np.zeros_like(self.C)
        _, my_kpt_inds = my_kpts_and_inds(np.arange(self.nkpts))
        for k in my_kpt_inds:
            C[k] = self.C[k] @ other.C[k]
        allreduce_inplace_contiguous(C)
        return MPIKBasisChanger(S, C, to_orthonormal=other.to_orthonormal)