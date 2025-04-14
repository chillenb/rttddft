from pyscf import lib
import numpy as np

import scipy.linalg as sla

def donothing(*args, **kwargs):
    pass


class MMUTPropagator:
    def __init__(self,
                 h1e,
                 dm,
                 bc,
                 get_veff,
                 v_ext,
                 all_mo_basis = False,
                 fock_init = None,
                 stepcallback = donothing,
                 starttime = 0.0,
                 dt = 0.1,
                 logger = None):

        self.h1e = h1e
        self.get_veff = get_veff
        self.bc = bc
        self.time = starttime
        self.dt = dt
        self.stepcallback = stepcallback
        self.v_ext = v_ext
        self.all_mo_basis = all_mo_basis

        if not all_mo_basis:
            fock_ao = fock_init if fock_init is not None else h1e + get_veff(dm=dm)
            self.fock = bc.rotate_focklike(fock_ao)
            self.dm = bc.rotate_denslike(dm)
            self.dm_min_half = self.dm.copy()
        else:
            self.fock = fock_init if fock_init is not None else h1e + get_veff(dm=dm)
            self.dm = dm
            self.dm_min_half = self.dm.copy()

        self.logger = logger

        self.stepcallback(self.time, self.dm)
        
    def step(self):
        F = self.fock
        nbuilds = 0

        Ft = F + self.v_ext(self.time)
        evs, evecs = sla.eigh(Ft)

        expw = evecs @ (np.exp(-1.0j * self.dt * evs)[:, None] * evecs.conj().T)
        expw_half = evecs @ (np.exp(-0.5j * self.dt * evs)[:, None] * evecs.conj().T)


        dm_p_half = np.linalg.multi_dot([expw, self.dm_min_half, expw.conj().T])

        dm_p_dt = np.linalg.multi_dot([expw_half, dm_p_half, expw_half.conj().T])



        if not self.all_mo_basis:
            dm_p_dt_ao = self.bc.rev_denslike(dm_p_dt)
            F_p_dt_ao = self.h1e + self.get_veff(dm=dm_p_dt_ao)
            F_p_dt = self.bc.rotate_focklike(F_p_dt_ao)
        else:
            F_p_dt = self.h1e + self.get_veff(dm=dm_p_dt)

        nbuilds += 1


        if self.logger is not None:
            self.logger.debug(f'MMUT: time {self.time:.3f}')



        self.dm = dm_p_dt
        self.dm_min_half = dm_p_half
        self.fock = F_p_dt
        self.time = self.time + self.dt
        self.stepcallback(self.time, self.dm)



