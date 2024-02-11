import numpy as np
from functools import reduce

def chaindot(*args):
    return reduce(np.dot, args)

class BasisChanger:
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
    def __init__(self, S, C, to_orthonormal=False):
        """
        Construct an instance of the basis change helper class.

        Args:
            S (np.ndarray): overlap matrix in original basis
            C (np.ndarray): transformation matrix.
            
            The columns of C are the new basis vectors expressed in the original basis.
            
        """
        self.S = S
        self.C = C
        self.to_orthonormal = to_orthonormal
        
        if to_orthonormal:
            self.Cinv = np.dot(C.conj().T, S)
        else:
            self.Cinv = np.linalg.inv(C)
        
    def rotate_focklike(self, mat):
        """
        Transform a Fock-like matrix from the original basis to the new basis.
        This is applicable to Fock, Hcore, Vj, Vk, etc.

        Args:
            mat (array_like): matrix to be transformed.

        Raises:
            ValueError: Dimension of mat must be >= 2.

        Returns:
            r_mat: rotated matrix (or array)
        """
        if mat.ndim >= 2:
            return chaindot(self.C.conj().T, mat, self.C)
        else:
            raise ValueError('mat.ndim must be >= 2')

    def rotate_denslike(self, mat):
        """
        Transform a density-like matrix from the original basis to the new basis.
        This is applicable to the density matrix, GF, etc.

        Args:
            mat (array_like): matrix to be transformed. Ignores axes > 2.

        Raises:
            ValueError: Dimension of mat must be >= 2.

        Returns:
            r_mat: rotated matrix (or array)
        """
        # for density matrix and GF
        if mat.ndim >= 2:
            return chaindot(self.Cinv, mat, self.Cinv.conj().T)
        else:
            raise ValueError('mat.ndim must be >= 2')

    def rotate_oplike(self, mat):
        # May not be needed.
        if mat.ndim >= 2:
            return chaindot(self.Cinv, mat, self.C)
        else:
            raise ValueError('mat.ndim must be >= 2')
    
    def rev_focklike(self, r_mat):
        """
        Transform a Fock-like matrix from the new basis to the original basis.
        This is applicable to Fock, Hcore, Vj, Vk, etc.

        Args:
            r_mat (array_like): matrix to be transformed. Ignores axes > 2.

        Raises:
            ValueError: Dimension of mat must be >= 2.

        Returns:
            mat: rotated matrix (or array)
        """
        # for Fock, Hcore, Vj, Vk, etc
        if r_mat.ndim >= 2:
            return chaindot(self.Cinv.conj().T, r_mat, self.Cinv)
        else:
            raise ValueError('rmat.ndim must be >= 2')

    def rev_denslike(self, r_mat):
        """
        Transform a density-like matrix from the new basis to the original basis.
        This is applicable to the density matrix, GF, etc.

        Args:
            r_mat (array_like): matrix to be transformed. Ignores axes > 2.

        Raises:
            ValueError: Dimension of mat must be >= 2.

        Returns:
            mat: rotated matrix (or array)
        """
        # for density matrix and GF
        if r_mat.ndim >= 2:
            return chaindot(self.C, r_mat, self.C.conj().T)
        else:
            raise ValueError('rmat.ndim must be >= 2')

    def rev_oplike(self, r_mat):
        # May not be needed.
        if r_mat.ndim >= 2:
            return chaindot(self.C, r_mat, self.Cinv)
        else:
            raise ValueError('rmat.ndim must be >= 2')
    
    
    
    def transform(self, mat, mat_type='focklike', rev=False):
        if not rev:
            if mat_type == 'focklike':
                return self.rotate_focklike(mat)
            elif mat_type == 'denslike':
                return self.rotate_denslike(mat)
            elif mat_type == 'oplike':
                return self.rotate_oplike(mat)
            else:
                raise ValueError(f'Unknown mat_type {mat_type}')
        else:
            if mat_type == 'focklike':
                return self.rev_focklike(mat)
            elif mat_type == 'denslike':
                return self.rev_denslike(mat)
            elif mat_type == 'oplike':
                return self.rev_oplike(mat)
            else:
                raise ValueError(f'Unknown mat_type {mat_type}')
        
    
    def inverse(self):
        if self.to_orthonormal:
            S_tilde = np.eye(self.S.shape[0])
        else:
            S_tilde = self.rotate_focklike(self.S)
        
        if np.linalg.norm(self.S - np.eye(self.S.shape[0])) < 1.0e-8:
            return BasisChanger(S_tilde, self.Cinv, to_orthonormal=True)
        return BasisChanger(S_tilde, self.Cinv)
    
    def chain(self, other):
        S = self.S
        C = self.C @ other.C
        return BasisChanger(S, C, to_orthonormal=other.to_orthonormal)
    
