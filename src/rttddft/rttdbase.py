import math
import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import symm
from pyscf.lib import logger
from pyscf.scf import hf_symm
from pyscf.scf import _response_functions
from pyscf.data import nist

from rttddft.propagators import magnus2
from rttddft.lib import BasisChanger

RTSCF_PROP_METHODS = {'magnus2'}

def gpulse_efield(t0, peak, sigma, dir=(0,0,1.0), freq=0.0, phaseshift=0.0):
    """
    Gaussian pulse electric field

    Parameters:
    t0 (float): The center time of the pulse.
    peak (float): The peak amplitude of the pulse.
    sigma (float): The width of the pulse.
    dir (tuple, optional): The direction of the electric field. Defaults to (0, 0, 1.0).
    freq (float, optional): The frequency of the sinusoidal component of the pulse. Defaults to 0.0.
    phaseshift (float, optional): The phase shift of the sinusoidal component of the pulse. Defaults to 0.0.

    Returns:
    function: A function that calculates the electric field at a given time.
    """
    uhat = np.asarray(dir)
    uhat = uhat / np.linalg.norm(uhat)
    
    def E(t):
        return (peak * np.exp( -(t-t0)**2 / (2*sigma**2) ) * \
                np.sin(freq*t + phaseshift)) * uhat
    return E

def kick_field(t0, peak, dir=(0,0,1.0)):
    uhat = np.asarray(dir)
    uhat = uhat / np.linalg.norm(uhat)
    def E(t):
        return (np.isclose(t,t0) * peak) * uhat
    return E


class RTTDSCF(lib.StreamObject):
    def __init__(self, mf, prop_method='magnus2'):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.max_memory = mf.max_memory

        self.wfnsym = None
        if prop_method not in RTSCF_PROP_METHODS:
            raise ValueError(f'prop_method {prop_method} not recognized')
        self.prop_method = prop_method
        
    
    def kernel(self, t_end, dt, t_start=0.0, efield=None):
        bc = BasisChanger(self._scf.get_ovlp(), self._scf.mo_coeff)
        log = logger.new_logger(self, self.verbose)
        with self.mol.with_common_origin((0,0,0)):
           ao_dip = self.mol.intor_symmetric('int1e_r', comp=3)
        mo_dip = bc.rotate_focklike(ao_dip)
        charges = self.mol.atom_charges()
        coords  = self.mol.atom_coords()
        nucl_dip = np.einsum('i,ix->x', charges, coords)

        if efield is not None:
            v_ext = lambda t: -np.einsum('xij,x->ij', mo_dip, efield(t))
        else: 
            v_ext = lambda t: 0.0
        
        self.trace = {'t': [], 'dipole': []}
        
        def stepcallback(t, dm):
            dipole = -np.sum(mo_dip * dm[None,:,:].real, axis=(1,2))
            self.trace['t'].append(t)
            self.trace['dipole'].append(dipole)
        

        if self.prop_method == 'magnus2':
            prop = magnus2.Magnus2Propagator(
                h1e = self.mol.intor('int1e_kin') + self.mol.intor('int1e_nuc'),
                get_veff = self._scf.get_veff,
                dm = self._scf.make_rdm1(),
                bc = bc,
                v_ext = v_ext,
                dt = dt,
                stepcallback=stepcallback,
                logger = log
            )
        else:
            raise ValueError(f'prop_method {self.prop_method} not recognized')
        
        if t_end <= t_start:
            raise ValueError('t_end must be greater than t_start')
        
        nsteps = math.ceil((t_end - t_start) / dt)
        
        for _ in range(nsteps):
            prop.step()
        self.prop = prop