import numpy as np
import scipy.linalg as sla
from rttddft.propagators.propstate import PropagatorState
from mpi4py import MPI
from pyscf import lib
from pyscf.pbc.mpitools.mpi_helper import allreduce_inplace_contiguous

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def purif(dm, restricted=True):
    e, u = sla.eigh(dm)
    e[np.abs(e)<1e-8] = 0.0
    occ = 2.0 if restricted else 1.0
    e[e>0.0] = occ
    mocc = u[:,e>0.0]
    return occ * (mocc@mocc.conj().T)

def step_magnus2_mo(state, h1e_mo, v_ext, S, get_veff_mo, dt, conv_tol=1e-5, bc=None,
    logger=None, callback=None, fock_ref=None, frozen=None, frozen_mask=None):
    converged = False
    nbuilds = 0
    dm = state.dm
    dm_prev = state.dm_prev
    F = state.fock
    F_m_dt = state.fock_prev
    F_p_half = 1.5 * F - 0.5 * F_m_dt

    if logger:
        logger.debug(f'F_m_dt vs F_p_half: {np.linalg.norm(F_m_dt-F_p_half):1.3e}')

    t = state.time
    F_p_dt = F_p_half

    if dm.ndim > 2:
        nkpts = dm.shape[0]
        is_kpoint = True
        my_kpt_inds = np.arange(rank, nkpts, size)
    else:
        nkpts = 0
        is_kpoint = False

    if dm_prev is not None:
        dm_p_dt = np.zeros_like(dm)
        if frozen is None:
            for k in my_kpt_inds:
                dm_p_dt[k] = purif(2.0 * dm[k] - dm_prev[k])
        else:
            for k in my_kpt_inds:
                frozen_mask_k = frozen_mask[k]
                mask2 = np.ix_(frozen_mask_k, frozen_mask_k)
                dm_p_dt[k] = dm[k]
                dm_p_dt[k][mask2] = purif(2.0 * dm[k][mask2] - dm_prev[k][mask2])
        allreduce_inplace_contiguous(comm, dm_p_dt)
    else:
        dm_p_dt = state.dm

    v_ext_half = v_ext(t + 0.5 * dt)
    if logger:
        logger.debug(f'v_ext_half: {np.linalg.norm(v_ext_half):1.3e}')

    adiis = lib.diis.DIIS(incore=True)

    while not converged:
        W = (F_p_half + v_ext_half)
        if is_kpoint:
            dm_p_dt_new = np.zeros_like(dm)
            if frozen is None:
                for k in my_kpt_inds:
                    evs, evecs = sla.eigh(W[k])
                    expw_k = evecs @ (np.exp(-1.0j * dt * evs)[:, None] * evecs.conj().T)
                    dm_p_dt_new[k] = purif(expw_k @ dm[k] @ expw_k.conj().T)
            else:
                for k in my_kpt_inds:
                    frozen_mask_k = frozen_mask[k]
                    mask2 = np.ix_(frozen_mask_k, frozen_mask_k)
                    evs, evecs = sla.eigh(W[k][mask2])
                    expw_k = evecs @ (np.exp(-1.0j * dt * evs)[:, None] * evecs.conj().T)
                    dm_p_dt_new[k][mask2] = purif(expw_k @ dm[k][mask2] @ expw_k.conj().T)
            allreduce_inplace_contiguous(comm, dm_p_dt_new)
        else:
            evs, evecs = sla.eigh(W)
            expw = evecs @ (np.exp(-1.0j * dt * evs)[:, None] * evecs.conj().T)
            dm_p_dt_new = expw @ dm @ expw.conj().T

        diff = np.linalg.norm(dm_p_dt_new - dm_p_dt)
        if is_kpoint:
            diff /= nkpts

        dm_p_dt = dm_p_dt_new

        if diff < conv_tol:
            converged = True
        else:
            if logger:
                logger.debug(f'Magnus2: diff={diff:1.3e}, conv_tol={conv_tol:1.3e}')

            # Direct MO evaluation without BasisChanger!
            F_p_dt = h1e_mo + get_veff_mo(dm_p_dt)

            F_p_dt = np.ascontiguousarray(adiis.update(F_p_dt))
            comm.Bcast(F_p_dt, root=0)

            nbuilds += 1
            F_p_half = 0.5 * (F + F_p_dt)

    if logger is not None:
        difference_norm = np.linalg.norm(dm_p_dt - dm)
        if is_kpoint:
            difference_norm /= nkpts
        logger.debug(f'Magnus2: time {t:.3f}, {nbuilds} get_veff call(s), |drho| = {difference_norm:1.3e}')

    new_state = PropagatorState(
        dm=dm_p_dt,
        dm_min_half=None,
        dm_prev=dm,
        fock=F_p_dt,
        fock_prev=F,
        time=t + dt,
        time_prev=t
    )

    if callback:
        callback(new_state)
    return new_state
