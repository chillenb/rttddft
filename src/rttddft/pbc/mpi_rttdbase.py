from functools import lru_cache
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import scf
from pyscf.pbc import df as pbcdf
from pyscf.pbc.gto import pseudo

from pyscf.pbc.df import gdf_builder, aft, rsdf_builder
from pyscf.pbc.df import gdf_builder, aft, mpi_rsdf_builder


from pyscf.pbc.df import rsdf
from pyscf.pbc.gto import pseudo
from pyscf.pbc.gto.pseudo.ppnl_velgauge import VelGaugePPNLHelper, get_gth_pp_nl_velgauge, get_gth_pp_nl_velgauge_commutator
from pyscf import __config__

from pyscf.pbc.mpitools.mpi_helper import allreduce_inplace_contiguous

# def allreduce_inplace_contiguous(comm, in_array):
#     if not in_array.flags.c_contiguous or not in_array.flags.c_contiguous:
#         raise ValueError("Input array must be contiguous")
#     view_1d = numpy.reshape(in_array, -1, order='A')
#     for i in range(0, view_1d.size, 2**30):
#         comm.Allreduce(MPI.IN_PLACE, in_array[i : min(i+2**30, view_1d.size)])



import math
import scipy

from pyscf.data import nist

from rttddft import rttdbase
from rttddft.rttdbase import make_vext_from_efield, get_mo_dip
from rttddft.propagators.propstate import PropagatorState
from rttddft.propagators import mpi_magnus2, mpi_mmut

from rttddft.lib.basischanger import KBasisChanger
from rttddft.lib.mpi_basischanger import MPIKBasisChanger


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def my_kpts_and_inds(kpts):
    my_kpt_inds = np.arange(rank, len(kpts), size)
    return kpts[my_kpt_inds], my_kpt_inds

def purif(dm, restricted=True):
    e, u = sla.eigh(dm)
    e[np.abs(e)<1e-8] = 0.0
    occ = 2.0 if restricted else 1.0
    if not np.allclose(e[e>0.0], occ):
        print(e)
        raise ValueError(f"occupancies are weird: {e}")
    e[e>0.0] = occ
    mocc = u[:,e>0.0]
    return occ * (mocc@mocc.conj().T)

RTSCF_PROP_METHODS = {'magnus2': mpi_magnus2.step_magnus2, 'mmut': mpi_mmut.step_mmut}

import h5py

def kick_afield(t0, F0, dir=(0,0,1.0)):
    # A(t) = -c F0 n theta(t)
    uhat = np.asarray(dir)
    uhat = uhat / np.linalg.norm(uhat)
    def E(t):
        return -1.0 * nist.LIGHT_SPEED * (np.sign(t-t0) * F0) * uhat
    return E

def zerofield(t):
    return (0.0,0.0,0.0)

def gaussian_afield(t0, sigma, F0, dir=(0,0,1.0)):
    uhat = np.asarray(dir)
    uhat = uhat / np.linalg.norm(uhat)
    def E(t):
        return -1.0 * nist.LIGHT_SPEED * scipy.special.erf((t - t0) / sigma) * F0 * uhat
    return E

def get_pseudopotential_local_part(mf, kpts=None):
    """Get the local part of the pseudopotential.
    Relevant for velocity-gauge calculations, where
    the nonlocal part of the pseudopotential depends on the vector potential
    and is frequently updated during time propagation.

    Works for AFT, GDF, and RSDF.

    Parameters
    ----------
    mf : PBC SCF object
    kpts : array_like, optional
        k-points, by default None

    Returns
    -------
    np.ndarray
        Local part of the pseudopotential.
    """
    cell = mf.cell
    if not cell.pseudo:
        raise RuntimeError('get_pseudopotential_local_part only works for pseudopotential calculations.')
    if mf.with_df is None:
        raise RuntimeError('get_pseudopotential_local_part only works for DF calculations.')
    with_df = mf.with_df
    cell = with_df.cell
    kpts, is_single_kpt = aft._check_kpts(with_df, kpts)

    if isinstance(with_df, pbcdf.GDF) or isinstance(mf.with_df, rsdf.RSDF):
        if with_df._prefer_ccdf or cell.omega > 0:
            raise NotImplementedError
        else:
            nuc_builder = rsdf_builder._RSNucBuilder(cell, kpts).build()
        t0 = (logger.process_clock(), logger.perf_counter())
        vpp_loc_part1 = nuc_builder.get_pp_loc_part1()
        t1 = logger.timer_debug1(nuc_builder, 'get_pp_loc_part1', *t0)
    elif isinstance(with_df, pbcdf.mpi_df.MPIGDF):
        nuc_builder = rsdf_builder._RSNucBuilder(cell, kpts).build()
        t0 = (logger.process_clock(), logger.perf_counter())
        vpp_loc_part1 = nuc_builder.get_pp_loc_part1()
        t1 = logger.timer_debug1(nuc_builder, 'get_pp_loc_part1', *t0)
    elif isinstance(with_df, aft.AFTDF):
        nuc_builder = with_df
        t0 = (logger.process_clock(), logger.perf_counter())
        vpp_loc_part1 = aft._get_pp_loc_part1(nuc_builder, kpts, with_pseudo=True)
        t1 = logger.timer_debug1(with_df, 'get_pp_loc_part1', *t0)
    else:
        raise NotImplementedError
    pp2builder = aft._IntPPBuilder(cell, kpts)
    vpp_loc_part_2 = pp2builder.get_pp_loc_part2()
    t2 = logger.timer_debug1(nuc_builder, 'get_pp_loc_part2', *t1)
    vpp_loc = vpp_loc_part1 + vpp_loc_part_2
    if is_single_kpt:
        vpp_loc = vpp_loc[0]
    return vpp_loc

def get_v_ext(cell, afield_t, kpts, S, h1e_ipovlp, bc=None, vgppnl_helper=None, ppnl_nofield=None):
    # q is -1 for electrons
    qA_over_c = -np.asarray(afield_t) / nist.LIGHT_SPEED
    logger.debug(cell, f'make_vext_velgauge: qA={qA_over_c}')
    my_kpts, my_kpt_inds = my_kpts_and_inds(kpts)

    h1e_ipovlp_my_k = h1e_ipovlp[my_kpt_inds]
    S_my_k = S[my_kpt_inds]
    qA_over_c_dot_p_my_k = np.einsum('i,kixy->kxy', qA_over_c, h1e_ipovlp_my_k) * (1.0j)

    qA_over_c_sqr = np.dot(qA_over_c, qA_over_c)
    if cell.pseudo:
        pp_nl = get_gth_pp_nl_velgauge(cell, q=qA_over_c, kpts=my_kpts, vgppnl_helper=vgppnl_helper)
        if ppnl_nofield is None:
            ppnl_nofield = get_gth_pp_nl_velgauge(cell, q=np.zeros(3), kpts=my_kpts, vgppnl_helper=vgppnl_helper)
        pp_nl -= ppnl_nofield
    else:
        pp_nl = 0.0
    nao = cell.nao_nr()
    vext_ao_local = qA_over_c_sqr * S_my_k - 2.0 * qA_over_c_dot_p_my_k + pp_nl
    vext_ao = np.zeros((len(kpts), nao, nao), dtype=np.complex128)
    vext_ao[my_kpt_inds] = vext_ao_local
    allreduce_inplace_contiguous(comm, vext_ao)

    if bc is not None:
        vext_mo = bc.rotate_focklike(vext_ao)
        return vext_mo
    else:
        return vext_ao

def make_vext_velgauge(cell, afield, kpts, S, h1e_ipovlp, bc=None, vgppnl_helper=None):
    @lru_cache(16)
    def v_ext_tup(afield_tuple):
        return get_v_ext(cell, afield_tuple, kpts, S, h1e_ipovlp, bc=bc, vgppnl_helper=vgppnl_helper)
    def v_ext(t):
        afield_t = afield(t)
        retval = v_ext_tup((float(afield_t[0]), float(afield_t[1]), float(afield_t[2])))
        # print(v_ext_tup.cache_info())
        return retval
    return v_ext

def get_electronic_velocity(cell, A, kpts, S, h1e_ipovlp, bc=None, dm=None, vgppnl_helper=None):
    qA_over_c = -1.0 * A / nist.LIGHT_SPEED
    my_kpts, my_kpt_inds = my_kpts_and_inds(kpts)
    h1e_ipovlp_my_k = h1e_ipovlp[my_kpt_inds]
    if cell.pseudo:
        with lib.temporary_env(cell, verbose=0):
            r_vnl_commutator = get_gth_pp_nl_velgauge_commutator(cell, q=qA_over_c, kpts=my_kpts, vgppnl_helper=vgppnl_helper)
    velocity = np.zeros(3, dtype=np.complex128)
    for k in range(len(my_kpts)):
        velocity += np.einsum('ixy,xy->i', h1e_ipovlp_my_k[k], dm[my_kpt_inds[k]], optimize=True) * (1.0j)
        if cell.pseudo:
            velocity += np.einsum('ixy,xy->i', r_vnl_commutator[k], dm[my_kpt_inds[k]], optimize=True) / (1.0j)
        velocity -= qA_over_c * np.einsum('xy,xy->', S[k], dm[my_kpt_inds[k]])
    allreduce_inplace_contiguous(comm, velocity)
    return velocity


def _frozen_sanity_check(frozen, mo_occ, kpt_idx):
    '''Performs a few sanity checks on the frozen array and mo_occ.

    Specific tests include checking for duplicates within the frozen array.

    Args:
        frozen (array_like of int): The orbital indices that will be frozen.
        mo_occ (:obj:`ndarray` of int): The occupation number for each orbital
            resulting from a mean-field-like calculation.
        kpt_idx (int): The k-point that `mo_occ` and `frozen` belong to.

    '''
    frozen = np.array(frozen)
    nocc = np.count_nonzero(mo_occ > 0)

    assert nocc, 'No occupied orbitals?\n\nnocc = %s\nmo_occ = %s' % (nocc, mo_occ)
    all_frozen_unique = (len(frozen) - len(np.unique(frozen))) == 0
    if not all_frozen_unique:
        raise RuntimeError('Frozen orbital list contains duplicates!\n\nkpt_idx %s\n'
                           'frozen %s' % (kpt_idx, frozen))
    if len(frozen) > 0 and np.max(frozen) > len(mo_occ) - 1:
        raise RuntimeError('Freezing orbital not in MO list!\n\nkpt_idx %s\n'
                           'frozen %s\nmax orbital idx %s' % (kpt_idx, frozen, len(mo_occ) - 1))

def get_frozen_mask(td):
    '''Boolean mask for orbitals in k-point post-HF method.

    Creates a boolean mask to remove frozen orbitals and keep other orbitals for post-HF
    calculations.

    Args:
        mp (:class:`MP2`): An instantiation of an SCF or post-Hartree-Fock object.

    Returns:
        moidx (list of :obj:`ndarray` of `bool`): Boolean mask of orbitals to include.

    '''
    moidx = [np.ones(x.size, dtype=bool) for x in td._scf.mo_occ]
    if td.frozen is None:
        pass
    elif isinstance(td.frozen, (int, np.integer)):
        for idx in moidx:
            idx[:td.frozen] = False
    elif isinstance(td.frozen[0], (int, np.integer)):
        frozen = list(td.frozen)
        for idx in moidx:
            idx[frozen] = False
    elif isinstance(td.frozen[0], (list, np.ndarray)):
        nkpts = len(td.frozen)
        if nkpts != td.nkpts:
            raise RuntimeError('Frozen list has a different number of k-points (length) than passed in mean-field/'
                               'correlated calculation.  \n\nCalculation nkpts = %d, frozen list = %s '
                               '(length = %d)' % (td.nkpts, td.frozen, nkpts))
        [_frozen_sanity_check(fro, mo_occ, ikpt) for ikpt, fro, mo_occ in zip(range(nkpts), td.frozen, td._scf.mo_occ)]
        for ikpt, kpt_occ in enumerate(moidx):
            kpt_occ[td.frozen[ikpt]] = False
    else:
        raise NotImplementedError

    return moidx

class MPIKRTTDSCF(rttdbase.RTTDSCF):
    _keys = {'cell', 'h1e_nuc_local', 'h1e_kin', 'h1e_ipovlp', 'vgppnl_helper'}

    get_frozen_mask = get_frozen_mask

    def __init__(self, mf, prop_method='magnus2', chkfile = None, frozen=None):
        super().__init__(mf, prop_method=prop_method, chkfile=chkfile)
        self.cell = mf.cell

        from pyscf.pbc.dft.multigrid import MultiGridNumInt
        if hasattr(mf, '_numint') and isinstance(mf._numint, MultiGridNumInt):
            raise NotImplementedError('Multigrid is not supported yet for RT-TDDFT')
        self.h1e_kin = None
        self.h1e_nuc_local = None
        self.h1e_ipovlp = None
        self.vgppnl_helper = None
        self.frozen = frozen
        self.nkpts = len(mf.kpts)

    def init_onebody_integrals(self):
        """Cache one-body integrals: kinetic, nuclear (local part of pseudopotentials if applicable),
           and <nabla mu | nu>.
        """
        mf = self._scf
        cell = self.cell
        kpts = mf.kpts
        my_kpts, my_kpt_inds = my_kpts_and_inds(kpts)
        nao = cell.nao_nr()

        self.h1e_nuc_local = np.zeros((len(kpts), nao, nao), dtype=np.complex128)
        if len(my_kpts) > 0:
            if cell.pseudo:
                h1e_nuc_local_my = get_pseudopotential_local_part(mf, my_kpts)
            else:
                h1e_nuc_local_my = mf.with_df.get_nuc(my_kpts)
            self.h1e_nuc_local[my_kpt_inds] = h1e_nuc_local_my
        allreduce_inplace_contiguous(comm, self.h1e_nuc_local)

        self.h1e_kin = np.zeros((len(kpts), nao, nao), dtype=np.complex128)
        if len(my_kpts) > 0:
            self.h1e_kin[my_kpt_inds] = np.asarray(cell.pbc_intor('int1e_kin', comp=1, hermi=1, kpts=my_kpts))
        allreduce_inplace_contiguous(comm, self.h1e_kin)
        
        self.h1e_ipovlp = np.zeros((len(kpts), 3, nao, nao), dtype=np.complex128)
        if len(my_kpts) > 0:
            self.h1e_ipovlp[my_kpt_inds] = np.asarray(cell.pbc_intor('int1e_ipovlp', comp=3, hermi=0, kpts=my_kpts))
        allreduce_inplace_contiguous(comm, self.h1e_ipovlp)




    def kernel(self, t_end, dt, t_start=0.0, efield=None, mo_basis=True, afield=None,
               conv_tol=1e-6):
        frozen = self.frozen

        self.init_onebody_integrals()
        kpts = self._scf.kpts
        nkpts = len(kpts)
        cell = self.cell
        local_kpts, local_kpt_inds = my_kpts_and_inds(kpts)

        dm = self._scf.make_rdm1()
        h1e = self.h1e_nuc_local + self.h1e_kin
        S = self._scf.get_ovlp()



        bc = KBasisChanger(self._scf.get_ovlp(), self._scf.mo_coeff, to_orthonormal=True, nkpts=nkpts)
        log = logger.new_logger(self, self.verbose)

        # with self.mol.with_common_origin((0.0, 0.0, 0.0)):
        #     ao_dip = self.mol.intor_symmetric('int1e_r', comp=3)
        # mo_dip = bc.rotate_focklike(ao_dip)
        # charges = self.mol.atom_charges()
        # coords  = self.mol.atom_coords()
        # nucl_dip = np.einsum('i,ix->x', charges, coords)

        
        self.trace = {'t': [], 'velocity': [], 'dm': []}

        if t_end <= t_start:
            raise ValueError('t_end must be greater than t_start')
        
        nsteps = math.ceil((t_end - t_start) / dt)




        if cell.pseudo:
            vgppnl_helper = VelGaugePPNLHelper(cell, local_kpts)
            vgppnl_helper.build()
        else:
            vgppnl_helper = None

        if rank == 0:
            chkf = h5py.File(self.chkfile, "w") if self.chkfile is not None else None
            if chkf is not None:
                chkf.create_dataset('t', (0,), maxshape=(None,), dtype=np.float64, chunks=True)
                chkf.create_dataset('velocity', (0, 3), maxshape=(None, 3), dtype=np.complex128, chunks=True)
                chkf.create_dataset('dm', (0, nkpts, self.mol.nao, self.mol.nao),
                                    dtype=np.complex128,
                                    maxshape=(None, nkpts, self.mol.nao, self.mol.nao),
                                    chunks=(1, nkpts, self.mol.nao, self.mol.nao))

        def stepcallback(state):
            t = state.time
            dm = state.dm
            dmao = bc.rev_denslike(dm)
            velocity = get_electronic_velocity(self.cell, afield(t), self._scf.kpts, S, self.h1e_ipovlp, dm=dmao, vgppnl_helper=self.vgppnl_helper)
            self.trace['t'].append(t)
            self.trace['velocity'].append(-velocity)
            self.trace['dm'].append(dm.copy())
            if rank == 0 and chkf is not None:
                chkf['t'].resize((chkf['t'].shape[0] + 1), axis=0)
                chkf['velocity'].resize((chkf['velocity'].shape[0] + 1), axis=0)
                chkf['dm'].resize((chkf['dm'].shape[0] + 1), axis=0)
                chkf['t'][-1] = t
                chkf['velocity'][-1] = np.asarray(velocity, dtype=np.complex128)
                chkf['dm'][-1] = np.asarray(dm, dtype=np.complex128)

        

        if self.prop is None:
            if self.prop_method in RTSCF_PROP_METHODS:
                self.prop = RTSCF_PROP_METHODS[self.prop_method]
            else:
                raise ValueError(f'prop_method {self.prop_method} not recognized')



        if hasattr(self._scf, '_numint'):
            def my_get_veff(dm_kpts):
                return self._scf.get_veff(dm=dm_kpts)
        else:
            def my_get_veff(dm_kpts):
                return self._scf.get_veff(dm_kpts=dm_kpts)

        if cell.pseudo:
            pp_nl_nofield = pseudo.pp_int.get_pp_nl(cell, kpts)
            pp_nl_nofield3 = get_gth_pp_nl_velgauge(cell, q=np.zeros(3), kpts=kpts, vgppnl_helper=self.vgppnl_helper)
            ppnl_err = np.linalg.norm(pp_nl_nofield-pp_nl_nofield3)
            if ppnl_err > 1e-5:
                raise ValueError(f"ppnl_err1={ppnl_err}")
        else:
            pp_nl_nofield = 0.0

        v_ext = make_vext_velgauge(cell, afield, kpts, S, self.h1e_ipovlp, bc=bc, vgppnl_helper=self.vgppnl_helper)

        veff = my_get_veff(dm_kpts=dm)
        fock_init = bc.rotate_focklike(h1e + veff + pp_nl_nofield)
        diag_err = 0
        offdiag_err = 0
        for k in range(nkpts):
            diag_err += np.linalg.norm(np.diag(fock_init[k])-self._scf.mo_energy[k])
            tmpmat = fock_init[k].copy()
            tmpmat -= np.diag(np.diag(tmpmat))
            offdiag_err += np.linalg.norm(tmpmat)
        if rank == 0:
            print(f"fock init diag error: {diag_err:1.3e}")
            print(f"fock init offdiag error: {offdiag_err:1.3e}")
            with h5py.File("fockerr.h5", "w") as outf:
                outf['h1e'] = h1e
                outf['veff'] = veff
                outf['h1e_nuc_local'] = self.h1e_nuc_local
                outf['h1e_kin'] = self.h1e_kin

            

        dm = np.asarray(
            [np.diag(self._scf.mo_occ[k]) for k in range(nkpts)],
            dtype=np.complex128
        )


        prop_state = PropagatorState(
                    dm = dm,
                    dm_min_half = dm,
                    dm_prev = None,
                    fock = fock_init,
                    fock_prev = fock_init,
                    time = t_start,
                    time_prev = t_start
                    )

        for _ in range(nsteps):
            prop_state = self.prop(
                state = prop_state,
                h1e = h1e + pp_nl_nofield,
                v_ext = v_ext,
                S = S,
                get_veff = my_get_veff,
                dt = dt,
                conv_tol = conv_tol,
                bc = bc,
                logger = log,
                callback = stepcallback,
                frozen=frozen,
                frozen_mask=self.get_frozen_mask()
            )
