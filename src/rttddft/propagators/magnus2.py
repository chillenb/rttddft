from pyscf import lib
import numpy as np

import scipy.linalg as sla

def donothing(*args, **kwargs):
    pass


class Magnus2Propagator:
    def __init__(self,
                 h1e,
                 dm,
                 bc,
                 get_veff,
                 v_ext,
                 fock_init = None,
                 fock_prev = None,
                 stepcallback = donothing,
                 starttime = 0.0,
                 dt = 0.1,
                 tol_interpol = 1e-7,
                 logger = None):
        
        self.h1e = h1e
        self.get_veff = get_veff
        self.bc = bc
        self.time = starttime
        self.dt = dt
        self.stepcallback = stepcallback
        self.tol_interpol = tol_interpol
        self.v_ext = v_ext
        
        fock_ao = fock_init if fock_init is not None else h1e + get_veff(dm=dm)
        
        self.fock = bc.rotate_focklike(fock_ao)
        
        fock_prev_ao = fock_prev if fock_prev is not None else self.fock
        
        self.fock_prev = bc.rotate_focklike(fock_prev_ao)
    
        self.dm = bc.rotate_denslike(dm)
    
        self.logger = logger
        
        self.stepcallback(self.time, self.dm)
        
    def step(self):
        F_m_half = self.fock_prev
        F = self.fock
        converged = False
        nbuilds = 0
        dm_p_dt = np.copy(self.dm)
        F_p_half = 2.0 * F - F_m_half
        F_p_dt = F_p_half

        while not converged:
            
            W = (-1.0j * self.dt) * (F_p_half + self.v_ext(self.time + 0.5 * self.dt))
            expw = sla.expm(W)
            
            dm_p_dt_new = np.linalg.multi_dot([expw, self.dm, expw.conj().T])
            
            diff = np.linalg.norm(dm_p_dt_new - dm_p_dt)
            
            dm_p_dt_old = dm_p_dt
            dm_p_dt = dm_p_dt_new
            
            if diff < self.tol_interpol:
                converged = True
            
            else:
                dm_p_dt_ao = self.bc.rev_denslike(dm_p_dt)
                F_p_dt_ao = self.h1e + self.get_veff(dm=dm_p_dt_ao)
                F_p_dt = self.bc.rotate_focklike(F_p_dt_ao)
                nbuilds += 1
                F_p_half = 0.5 * (F + F_p_dt)
                        
            if self.logger is not None:
                self.logger.debug(f'Magnus2: time {self.time:.3f}, {nbuilds} get_veff call(s), |drho| = {diff:1.3e}')
            


        self.dm = dm_p_dt
        self.fock_prev = F
        self.fock = F_p_dt
        self.time = self.time + self.dt
        self.stepcallback(self.time, self.dm)
        