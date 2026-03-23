import numpy as np
import scipy
from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos

from pyscf.pbc.mpitools.mpi_helper import allreduce_inplace_contiguous

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def partition_range(L, n):
    L_div_n, L_mod_n = divmod(L, n)
    chunksizes = np.zeros(n + 1, dtype=int)
    chunksizes[1:L_mod_n + 1] = L_div_n + 1
    chunksizes[L_mod_n + 1:] = L_div_n
    return np.cumsum(chunksizes)

def get_subrange(L, n, k):
    divpts = partition_range(L, n)
    return int(divpts[k]), int(divpts[k + 1])

def global_to_local_inds(divpts, global_i):
    my_divpt = int(np.searchsorted(divpts, global_i, side='right')) - 1
    return my_divpt, global_i - int(divpts[my_divpt])

class DistDiel:
    def __init__(self, mf, kpts, qp_energies):
        self.mf = mf
        self.mo_coeff = np.asarray(mf.mo_coeff)
        self.kpts = kpts
        self.nocc = int(mf.cell.nelectron // 2)
        self.nkpts = len(kpts)
        self.qp_energies = qp_energies



        # ndarray[int] describing global distribution of k-points
        # length: size + 1
        # rank p owns k_partition_divpts[p] -- k_partition_divpts[p+1].
        self.k_partition_divpts = None

        # ndarray[int] containing k_partition_divpts[p] -- k_partition_divpts[p+1].
        self.kL_inds = None

        # Used to partition k-point pairs over all ranks.
        self.kpts_i = None
        self.kpts_j = None
        self.kpts_L = None

        # cholesky factor of I - Pi[kL]
        self.diel_cho = None

        # MO-transformed CDERIs.
        self.cderiarr_slice = None

        # k-diagonal of MO-transformed CDERI. Used for get-j.
        self.cderiarr_diag_slice = None

        # chol(I - Pi[kL])^-1 Lpq[ki, kj], same distribution
        # as self.cderiarr_slice.
        self.screened_cderiarr_slice = None

        # Pi[kL].
        self.Pi_static = None
        # Pi[kL] ref vals used for debugging
        self.Pi_static_ref = None

        self.calc_kconserv()

    def calc_kconserv(self):
        kpts = self.kpts
        nkpts = len(kpts)
        self.k_partition_divpts = partition_range(nkpts, size)
        self.kL_inds = np.arange(*get_subrange(nkpts, size, rank))

        if hasattr(self.mf, 'cell') and hasattr(self.mf.cell, 'get_scaled_kpts'):
            kscaled = self.mf.cell.get_scaled_kpts(kpts)
            kscaled -= kscaled[0]
        else:
            kscaled = kpts

        kpts_i = []
        kpts_j = []
        kpts_L = []

        for kL in self.kL_inds:
            for i, kpti in enumerate(kpts):
                for j, kptj in enumerate(kpts):
                    kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                    is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                    if is_kconserv:
                        kpts_i.append(i)
                        kpts_j.append(j)
                        kpts_L.append(kL)
        self.kpts_i = np.asarray(kpts_i, dtype=int)
        self.kpts_j = np.asarray(kpts_j, dtype=int)
        self.kpts_L = np.asarray(kpts_L, dtype=int)
        self.nkpt_pairs = len(self.kpts_i)


    def load_all(self):
        mo_coeff = self.mf.mo_coeff
        naux = self.mf.with_df.get_naoaux()
        nao = self.mf.cell.nao
        nmo = nao
        kpts = self.kpts
        nkpts = len(kpts)
        win_shape = (self.nkpt_pairs, naux, nmo, nmo)

        self.cderiarr_slice = np.empty(
            dtype=np.complex128,
            shape=win_shape
        )

        cderiarr = self.mf.with_df.cderi_array()
        for ij in range(self.nkpt_pairs):
            i = self.kpts_i[ij]
            j = self.kpts_j[ij]
            kpti = self.kpts[i]
            kptj = self.kpts[j]
            Lpq = cderiarr.load(kpti, kptj)
            if Lpq.shape[-1] == (nao*(nao+1))//2:
                Lpq = lib.unpack_tril(Lpq).reshape(-1,nao**2)
            else:
                Lpq = Lpq.reshape(-1,nao**2)
            Lpq = Lpq.astype(np.complex128)
            moij, ijslice = _conc_mos(mo_coeff[i], mo_coeff[j])[2:]
            _ao2mo.r_e2(Lpq, moij, ijslice, tao=[], ao_loc=None,
                out=self.cderiarr_slice[ij]
            )

        # Distribute the k-diagonal of Lpq across all ranks. The k-diagonal is always
        # stored on rank 0.
        self.cderiarr_diag_slice = np.zeros(
            dtype=np.complex128,
            shape=(len(self.kL_inds), naux, nmo, nmo)
        )

        reqs = []
        if rank == 0:
            kdiag_distribution = partition_range(nkpts, size)
            root_kdiag_inds = np.flatnonzero(self.kpts_i == self.kpts_j)
            for lidx in root_kdiag_inds:
                ki = self.kpts_i[lidx]
                kj = self.kpts_j[lidx]
                assert ki == kj
                target_rank, target_slot = global_to_local_inds(kdiag_distribution, ki)
                Lia = self.cderiarr_slice[lidx]
                if target_rank == 0:
                    self.cderiarr_diag_slice[target_slot] = Lia
                else:
                    reqs.append(comm.Isend(Lia, dest=target_rank, tag=111 + ki))
        else:
            for idx, ki in enumerate(self.kL_inds):
                reqs.append(comm.Irecv(self.cderiarr_diag_slice[idx], source=0, tag=111 + ki))
                
        MPI.Request.Waitall(reqs)



    def get_static_diel(self):
        mo_energy = self.qp_energies
        nocc = int(self.mf.cell.nelectron // 2)
        naux = self.mf.with_df.get_naoaux()
        nao = self.mf.cell.nao
        nmo = nao
        nvir = nmo - nocc
        nkpts = self.nkpts

        self.Pi_static = np.zeros((len(self.kL_inds), naux, naux), dtype=np.complex128)

        for ij, (kL, ki, ka) in enumerate(zip(self.kpts_L, self.kpts_i, self.kpts_j)):
            ikL = kL - self.k_partition_divpts[rank]
            Pi = self.Pi_static[ikL]
            # Find ka that conserves with ki and kL (-ki+ka+kL=G)
            Lia_i = np.ascontiguousarray(self.cderiarr_slice[ij][:, :nocc, nocc:])
            sqrteia = np.sqrt(mo_energy[ka][None, nocc:] - mo_energy[ki][:nocc, None])

            rsqrteia = (1.0 / sqrteia).astype(Lia_i.dtype)
            Pia = lib.broadcast_mul(Lia_i, rsqrteia)

            # Since trans=2, C=a^H a
            #                 = Pia.reshape(naux, nocc * nvir).T.conj() 
            #                    @ Pia.reshape(naux, nocc * nvir)
            #                 = einsum('Qia, Pia->QP', Pia.conj(), Pia)
            # With C = Pi.T,
            # we have Pi = np.einsum('Pia, Qia->PQ', Pia, Pia.conj())
            scipy.linalg.blas.zherk(
                alpha=-4.0 / nkpts,
                a=Pia.reshape(naux, nocc * nvir).T,
                c=Pi.T,
                trans=2,
                beta=1.0,
                overwrite_c=True,
            )
        for ikL in range(len(self.kL_inds)):
            lib.hermi_triu(self.Pi_static[ikL], inplace=True)

    def get_screened_Lpq(self):
        naux = self.mf.with_df.get_naoaux()
        nao = self.mf.cell.nao
        nmo = nao

        self.screened_cderiarr_slice = np.empty(
            dtype=np.complex128,
            shape=(self.nkpt_pairs, naux, nmo, nmo)
        )

        self.diel_cho = np.empty(
            dtype=np.complex128,
            shape=(len(self.kL_inds), naux, naux)
        )

        for ikL, kL in enumerate(self.kL_inds):
            Pi = self.Pi_static[ikL]
            self.diel_cho[ikL] = scipy.linalg.cholesky(np.eye(naux) - Pi, overwrite_a=True, lower=True)

        for ij, (kL, ki, kj) in enumerate(zip(self.kpts_L, self.kpts_i, self.kpts_j)):
            ikL = kL - self.k_partition_divpts[rank]
            Pi_chol = self.diel_cho[ikL]
            Lpq = self.cderiarr_slice[ij]
            self.screened_cderiarr_slice[ij] = scipy.linalg.solve_triangular(
                Pi_chol, Lpq.reshape(naux, -1), lower=True, check_finite=False).reshape(naux, nmo, nmo)

    def get_j(self, dm_kpts):
        nkpts = self.nkpts
        naux = self.mf.with_df.get_naoaux()
        nmo = self.mf.cell.nao
        
        v_j = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
        rho_P = np.zeros(naux, dtype=np.complex128)

        # Each rank calculates the partial rho_P from its assigned k points
        for idx, k in enumerate(self.kL_inds):
            Lpq = self.cderiarr_diag_slice[idx]
            rho_P += np.einsum('Pij,ji->P', Lpq, dm_kpts[k], optimize=True)
            
        allreduce_inplace_contiguous(comm, rho_P)
        
        rho_P *= (1.0 / nkpts)
        
        # Each rank calculates the partial v_j for its assigned k points
        for idx, k in enumerate(self.kL_inds):
            Lpq = self.cderiarr_diag_slice[idx]
            v_j[k] = np.einsum('Pij,P->ij', Lpq, rho_P, optimize=True)
                
        allreduce_inplace_contiguous(comm, v_j)
        
        return v_j

    def get_k(self, dm_kpts=None, mo_coeff=None, screened=False):
        kpts = self.kpts
        nkpts = len(kpts)
        local_k_inds = np.arange(*get_subrange(nkpts, size, rank))
        nkpts = self.nkpts
        naux = self.mf.with_df.get_naoaux()
        nmo = self.mf.cell.nao
        nocc = int(self.mf.cell.nelectron // 2)

        if dm_kpts is None and mo_coeff is None:
            raise ValueError("One of dm_kpts or mo_coeff must be provided.")

        if dm_kpts is not None and mo_coeff is not None:
            raise ValueError("Cannot provide both dm_kpts and mo_coeff.")

        if dm_kpts is not None:
            mo_coeff_sqrtocc = np.zeros((nkpts, nmo, nocc), dtype=np.complex128)
            tol = 1e-8
            for k in local_k_inds:
                e, u = scipy.linalg.eigh(dm_kpts[k])
                e, u = e[e>tol], u[:, e>tol]
                mo_coeff_sqrtocc[k] = u[:, :nocc] * np.sqrt(2)
            allreduce_inplace_contiguous(comm, mo_coeff_sqrtocc)

        else:
            mo_coeff_sqrtocc = np.ascontiguousarray(
                [mo_coeff[k][:, :nocc] * np.sqrt(2) for k in range(nkpts)]
            )
        
        # mo_coeff is now (nkpts, nmo, nocc) C(p,i).

        v_k = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)


    # K(p,q; k2 from k1)
    # --> in case of Hermitian & PSD DM
    #     = ( V(L, s k1, p k2) * C(s,i; k1).conj() ).conj()
    #       * V(L, r k1, q k2) * C(r,i; k1).conj()                          eqn (2)
    #     = W(L, i k1, p k2).conj() * W(L, i k1, q k2)                      eqn (3)

        iWq_flat = np.zeros((nocc, naux * nmo), dtype=np.complex128)
        for ij, (ki, kj) in enumerate(zip(self.kpts_i, self.kpts_j)):
            # We update v_k[kj] here.
            if not screened:
                Lpq = self.cderiarr_slice[ij]
            else:
                Lpq = self.screened_cderiarr_slice[ij]
            pLq = np.ascontiguousarray(Lpq.transpose(1,0,2))
            Cri_ki = mo_coeff_sqrtocc[ki]

            pLq_flat = pLq.reshape(nmo, naux * nmo)

            # iWq = np.einsum('rLq,ri->iLq', pLq, Cri_ki.conj(), optimize=True)
            scipy.linalg.blas.zgemm(
                alpha=1.0,
                a=pLq_flat.T, # (naux*nmo, nmo) qLr
                b=Cri_ki.T,   # (nocc, nmo) ir
                c=iWq_flat.T, # (naux*nmo, nocc) iLq
                trans_a=0,
                trans_b=2,
                overwrite_c=True,
            )

            iWq_flat2 = iWq_flat.reshape(nocc * naux, nmo)

            # v_k[kj] += np.einsum('iLp,iLq->pq', iWq, iWq.conj(), optimize=True)
            scipy.linalg.blas.zherk(
                alpha=1.0,
                a=iWq_flat2.T, # (nmo, naux * nocc) qLi
                c=v_k[kj].T, # (nmo, nmo) qp
                trans=0,
                beta=1.0,
                overwrite_c=True,
            )
        for k in range(nkpts):
            v_k[k] = lib.hermi_triu(v_k[k].conj()) / nkpts
        allreduce_inplace_contiguous(comm, v_k)
        return v_k

    def get_screened_k(self, dm_kpts=None, mo_coeff=None):
        return self.get_k(dm_kpts=dm_kpts, mo_coeff=mo_coeff, screened=True)

    def get_static_diel_ref(self):
        mo_energy = self.qp_energies
        nocc = int(self.mf.cell.nelectron // 2)
        naux = self.mf.with_df.get_naoaux()
        nkpts = self.nkpts

        self.Pi_static_ref = np.zeros((len(self.kL_inds), naux, naux), dtype=np.complex128)

        for ia, (kL, ki, ka) in enumerate(zip(self.kpts_L, self.kpts_i, self.kpts_j)):
            ikL = kL - self.k_partition_divpts[rank]
            Pi = self.Pi_static_ref[ikL]
            # Find ka that conserves with ki and kL (-ki+ka+kL=G)
            Lia_i = np.ascontiguousarray(self.cderiarr_slice[ia][:, :nocc, nocc:])
            eia = mo_energy[ki][:nocc, None] - mo_energy[ka][None, nocc:]

            # compare expressions from RPA and GW, with omega=0.
            eia = (1.0 / eia).astype(Lia_i.dtype)
            Pia = lib.broadcast_mul(Lia_i, eia)
            # Response from both spin-up and spin-down density
            Pi += (4./nkpts) * lib.einsum('Pia,Qia->PQ', Pia, Lia_i.conj())

    def get_screened_k_ref(self, dm_kpts=None, mo_coeff=None, screened=True, strategy=1):
        kpts = self.kpts
        nkpts = len(kpts)
        local_k_inds = np.arange(*get_subrange(nkpts, size, rank))
        nkpts = self.nkpts
        naux = self.mf.with_df.get_naoaux()
        nmo = self.mf.cell.nao
        nocc = int(self.mf.cell.nelectron // 2)

        if dm_kpts is None and mo_coeff is None:
            raise ValueError("One of dm_kpts or mo_coeff must be provided.")

        if dm_kpts is not None and mo_coeff is not None:
            raise ValueError("Cannot provide both dm_kpts and mo_coeff.")

        if dm_kpts is not None:
            mo_coeff_sqrtocc = np.zeros((nkpts, nmo, nocc), dtype=np.complex128)
            tol = 1e-8
            for k in local_k_inds:
                e, u = scipy.linalg.eigh(dm_kpts[k])
                e, u = e[e>tol], u[:, e>tol]
                mo_coeff_sqrtocc[k] = u[:, :nocc] * np.sqrt(2)
            allreduce_inplace_contiguous(comm, mo_coeff_sqrtocc)
            
        else:
            mo_coeff_sqrtocc = np.ascontiguousarray(
                [mo_coeff[k][:, :nocc] * np.sqrt(2) for k in range(nkpts)]
            )
            # dm_kpts = np.asarray([mo_coeff_sqrtocc[k][:, :nocc] @ mo_coeff_sqrtocc[k][:, :nocc].conj().T for k in range(nkpts)])

        # if rank == 0:
        #     print(dm_kpts)

        v_k = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
    # K(p,q; k2 from k1)
    # --> in case of Hermitian & PSD DM
    #     = ( V(L, s k1, p k2) * C(s,i; k1).conj() ).conj()
    #       * V(L, r k1, q k2) * C(r,i; k1).conj()                          eqn (2)
    #     = W(L, i k1, p k2).conj() * W(L, i k1, q k2)                      eqn (3)
        for ij, (kL, ki, kj) in enumerate(zip(self.kpts_L, self.kpts_i, self.kpts_j)):
            ikL = kL - self.k_partition_divpts[rank]

            Lpq = self.cderiarr_slice[ij]
            Pi = self.Pi_static_ref[ikL]

            # calculate the inverse dielectric function
            if screened == True:
                InvD = np.linalg.inv((np.eye(naux) - Pi))
            else:
                InvD = np.eye(naux)

            if strategy == 1: # symm mo coeff
            # calculate the auxiliary matrix
                if screened == True:
                    Lpq_bar = lib.einsum('PQ,Qmn->Pmn', InvD, Lpq)
                    intermediate1 = lib.einsum('Lsp, si -> Lip', Lpq_bar, mo_coeff_sqrtocc[ki].conj())
                else:
                    intermediate1 = lib.einsum('Lsp, si -> Lip', Lpq, mo_coeff_sqrtocc[ki].conj())                
                intermediate2 = lib.einsum('Lrq, ri -> Liq', Lpq, mo_coeff_sqrtocc[ki].conj())
                v_k[kj] += lib.einsum('Lip, Liq->pq', intermediate1.conj(), intermediate2)

            elif strategy == 2: # dm build
                if screened:
                    v_k[kj] += lib.einsum('Psp, Qrq, sr, PQ->pq', Lpq.conj(), Lpq, dm_kpts[ki], InvD)
                else:
                    v_k[kj] += lib.einsum('Lsp, Lrq, sr->pq', Lpq.conj(), Lpq, dm_kpts[ki])
            else:
                raise ValueError("asymm mo coeff not implemented")

        for k in range(nkpts):
            v_k[k] /= (nkpts)

        allreduce_inplace_contiguous(comm, v_k)
        return v_k