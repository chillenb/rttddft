from functools import lru_cache
import math
import scipy
import numpy as np
import h5py

from pyscf import lib
from pyscf.data import nist
from pyscf.lib import logger
from pyscf.pbc import scf
from pyscf.pbc import df as pbcdf
from pyscf.pbc.gto import pseudo

from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos

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








from rttddft import rttdbase
from rttddft.rttdbase import make_vext_from_efield, get_mo_dip
from rttddft.propagators.propstate import PropagatorState
from rttddft.propagators import mpi_magnus2, mpi_mmut

from rttddft.lib.basischanger import KBasisChanger
from rttddft.lib.mpi_basischanger import MPIKBasisChanger
from rttddft.propagators.mpi_magnus2_mo import step_magnus2_mo
from rttddft.pbc.har_ex import DistDiel


from rttddft.pbc.mpi_rttdbase import my_kpts_and_inds, \
    purif, kick_afield, RTSCF_PROP_METHODS, \
    zerofield, gaussian_afield, get_pseudopotential_local_part, \
    get_v_ext, make_vext_velgauge, get_electronic_velocity, \
    _frozen_sanity_check, get_frozen_mask, MPIKRTTDSCF


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class MPIKTDAGW(MPIKRTTDSCF):

    def __init__(self, mf, prop_method='magnus2_mo', chkfile = None, frozen=None,
                 qp_energies = None):
        super().__init__(mf, prop_method=prop_method, chkfile=chkfile, frozen=frozen)
        if qp_energies is None:
            raise ValueError("qp_energies must be provided")
        self.qp_energies = np.asarray(qp_energies)


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



        bc = MPIKBasisChanger(self._scf.get_ovlp(), self._scf.mo_coeff, to_orthonormal=True, nkpts=nkpts)
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
            if self.prop_method == 'magnus2_mo':
                self.prop = step_magnus2_mo
            elif self.prop_method in RTSCF_PROP_METHODS:
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

        dm = np.asarray(
            [np.diag(self._scf.mo_occ[k]) for k in range(nkpts)],
            dtype=np.complex128
        )

        h1e_mo = np.zeros((nkpts, self.mol.nao, self.mol.nao), dtype=np.complex128)
        for k in range(nkpts):
            h1e_mo[k] = np.diag(self.qp_energies[k])
            
        dd = DistDiel(self._scf, kpts, self.qp_energies)
        dd.load_all()
        dd.get_static_diel()
        
        vz0 = dd.get_j(dm)
        waz0 = -1.0 * dd.get_screened_k(dm_kpts=dm)

        def get_veff_mo(dm_mo):
            vz = dd.get_j(dm_mo)
            waz = -1.0 * dd.get_screened_k(dm_kpts=dm_mo)
            return (vz - vz0) + 0.5 * (waz - waz0)
            
        fock_init = h1e_mo.copy() # Starts exactly at QP energies (veff_mo initialized to zero)


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
                h1e_mo = h1e_mo,
                v_ext = v_ext,
                S = S,
                get_veff_mo = get_veff_mo,
                dt = dt,
                conv_tol = conv_tol,
                bc = bc,
                logger = log,
                callback = stepcallback,
                frozen=frozen,
                frozen_mask=self.get_frozen_mask()
            )
