import numpy as np
import scipy.linalg
from pyscf.pbc import gto, scf, df
from pyscf import lib
import sys
import os

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

from rttddft.pbc.har_ex import DistDiel

def run_test():
    # 1. Setup PBC Cell
    cell = gto.Cell()
    cell.atom = 'He 0 0 0'
    cell.basis = 'gth-dzvp'
    cell.pseudo = 'gth-pade'
    cell.a = np.eye(3) * 3.0
    cell.verbose = 3
    cell.build()

    # 2. 2x2x2 k-mesh
    kpts = cell.make_kpts([2, 2, 2])
    nkpts = len(kpts)

    # 3. Setup mean-field
    mf = scf.KRHF(cell, kpts)
    mf.with_df = df.GDF(cell, kpts)
    mf.with_df._cderi = 'he3_gdf.h5'
    mf.exxdiv = None

    data = lib.chkfile.load('he3.chk', 'scf')
    mf.__dict__.update(data)

    dm_ao = mf.make_rdm1()
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    dm_mo = np.array([np.diag(occ) for occ in mo_occ])

    vj_ao, vk_ao = mf.get_jk(cell, dm_ao, kpts=kpts)
    vj_pyscf_mo = np.array([C.conj().T @ vj @ C for C, vj in zip(mo_coeff, vj_ao)])
    vk_pyscf_mo = np.array([C.conj().T @ vk @ C for C, vk in zip(mo_coeff, vk_ao)])

    qp_energies = mf.mo_energy
    dd = DistDiel(mf, kpts, qp_energies)
    dd.load_all()

    for p in range(size):
        comm.Barrier()
        if p == rank:
            print(f"rank: {rank}, kL_inds: {dd.kL_inds}")
            print(f"rank: {rank}, kpts_i: {dd.kpts_i}")
            print(f"rank: {rank}, kpts_j: {dd.kpts_j}")
            print(f"rank: {rank}, cderiarr_diag_slice: {dd.cderiarr_diag_slice.shape}")


    vj_dist_mo = dd.get_j(dm_mo)
    mo_basis_coeffs = [np.eye(cell.nao) for _ in range(nkpts)]
    vk_dist_mo = dd.get_k(dm_kpts=dm_mo)


    diff_j = np.max(np.abs(vj_pyscf_mo - vj_dist_mo))
        
    diff_k = np.max(np.abs(vk_pyscf_mo - vk_dist_mo))

    diff_j = comm.reduce(diff_j, op=MPI.MAX)
    diff_k = comm.reduce(diff_k, op=MPI.MAX)

    if rank == 0:
        print(f"Max diff in J (MO basis): {diff_j:.2e}")
        print(f"Max diff in K (MO basis): {diff_k:.2e}")


    dd.get_static_diel_ref()
    dd.get_static_diel()

    norm_pi_sq = np.linalg.norm(dd.Pi_static)**2
    norm_pi_sq = comm.reduce(norm_pi_sq)


    diff_static_diel = np.max(np.abs(dd.Pi_static_ref - dd.Pi_static))
    diff_static_diel = comm.reduce(diff_static_diel, op=MPI.MAX)
    if rank == 0:
        norm_pi = np.sqrt(norm_pi_sq)
        print(f"Max diff in Pi: {diff_static_diel:.2e}")
        print(f"||Pi||: {norm_pi:.2e}")

    dd.get_screened_Lpq()
    screened_k_ref = dd.get_screened_k_ref(dm_kpts=dm_mo)
    screened_k = dd.get_screened_k(dm_kpts=dm_mo)

    k_ref = dd.get_screened_k_ref(dm_kpts=dm_mo, screened=False)

    diff_screened_k = np.max(np.abs(screened_k - screened_k_ref))
    diff_screened_k = comm.reduce(diff_screened_k, op=MPI.MAX)
    diff_kre = np.max(np.abs(vk_dist_mo.real - k_ref.real))
    diff_kre = comm.reduce(diff_kre, op=MPI.MAX)
    diff_kim = np.max(np.abs(vk_dist_mo.imag - k_ref.imag))
    diff_kim = comm.reduce(diff_kim, op=MPI.MAX)
    if rank == 0:
        print(f"Max diff in screened_k: {diff_screened_k:.2e}")
        print(f"Max diff in k (re): {diff_kre:.2e}")
        print(f"Max diff in k (im): {diff_kim:.2e}")
        # print(dm_mo)

if __name__ == '__main__':
    run_test()
