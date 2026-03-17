import numpy as np

import scipy.linalg as sla
from rttddft.propagators.propstate import PropagatorState

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from pyscf.pbc.mpitools.mpi_helper import allreduce_inplace_contiguous


def mcweeny(dm, restricted=True):
    dm_id = 0.5 * dm if restricted else dm
    dm2 = dm_id @ dm_id
    dm_pure = 3.0 * dm2 - 2.0 * (dm_id @ dm2)
    retval = 2.0 * dm_pure if restricted else dm_pure
    return retval


def purif(dm, restricted=True):
    e, u = sla.eigh(dm)
    e[np.abs(e)<1e-8] = 0.0
    occ = 2.0 if restricted else 1.0
    e[e>0.0] = occ
    mocc = u[:,e>0.0]
    return occ * (mocc@mocc.conj().T)



def step_magnus2(state, h1e, v_ext, S, get_veff, dt, conv_tol=1e-5, bc=None,
    logger=None, callback=None, fock_ref=None, frozen=None, frozen_mask=None):
    """Perform a single predictor/corrector time step using the Magnus expansion.

    Parameters
    ----------
    state : PropagatorState
        Current system state.
    h1e : np.ndarray
        Time-independent part of the one-electron Hamiltonian.
    v_ext : function
        Function returning the external potential at a given time.
    get_veff : function
        Function mapping the density matrix to the effective potential.
    dt : float
        Time step length
    conv_tol : float, by default 1e-5
        Convergence tolerance for the predictor/corrector step.
    bc : BasisChanger, optional
        basis changer for MO basis; only needed if mo_basis is True, by default None
    logger : pyscf.lib.logger, optional
        logger object, by default None
    callback : function, optional
        function to call after each time step, by default None.
        Invoked as callback(new_state), where new_state is of type PropagatorState.

    Returns
    -------
    PropagatorState
        New system state after the time step.
    """
    converged = False
    nbuilds = 0
    dm = state.dm
    dm_prev = state.dm_prev
    F = state.fock
    F_m_dt = state.fock_prev
    F_p_half = 1.5 * F - 0.5 * F_m_dt

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
            frozen_mask_k = frozen_mask[k]
            mask2 = np.ix_(frozen_mask_k, frozen_mask_k)
            for k in my_kpt_inds:
                dm_p_dt[k] = dm[k]
                dm_p_dt[k][mask2] = purif(2.0 * dm[k][mask2] - dm_prev[k][mask2])
        allreduce_inplace_contiguous(comm, dm_p_dt)
    else:
        dm_p_dt = state.dm

    v_ext_half = v_ext(t + 0.5 * dt)
    logger.debug(f'v_ext_half: {np.linalg.norm(v_ext_half):1.3e}')
    while not converged:

        # nondiag_norm = 0.0
        # for k in my_kpt_inds:
        #     nondiag_norm += np.linalg.norm(
        #         dm_p_dt[k] - np.diag(np.diag(dm_p_dt[k]))
        #     )
        # nondiag_norm = comm.allreduce(nondiag_norm)
        # logger.debug(f'nondiag_norm: {nondiag_norm:1.4e}')
        # if rank == 0:
        #     logger.debug(f'{np.diag(dm_p_dt[0])}')

        W = (F_p_half + v_ext_half)
        # k-point case
        # todo: MPI parallelization
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
            logger.debug(f'Magnus2: diff={diff:1.3e}, conv_tol={conv_tol:1.3e}')

            assert bc is not None, "BasisChanger 'bc' must be provided to define the MO basis"
            dm_p_dt_ao = bc.rev_denslike(dm_p_dt)
            F_p_dt_ao = h1e + get_veff(dm_p_dt_ao)
            F_p_dt = bc.rotate_focklike(F_p_dt_ao)


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


