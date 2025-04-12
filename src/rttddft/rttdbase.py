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

import h5py

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

def sine_efield(E0, omega, dir=(0,0,1.0), start=0.0, shift=0.0):
    uhat = np.asarray(dir)
    uhat = uhat / np.linalg.norm(uhat)
    def E(t):
        if t < start:
            return 0.0 * uhat
        else:
            return E0 * np.sin(omega * (t - start) - shift) * uhat
    return E

def intensity_to_e0_au(intensity):
    """Convert intensity in W/cm^2 to E0 in atomic units

    Parameters
    ----------
    intensity : float
        Intensity in W/cm^2
    
    Returns
    -------
    float
        E0 in atomic units
    """
    return np.sqrt(intensity / (3.50944758e16))

def make_bc(mf):
    return BasisChanger(mf.get_ovlp(), mf.mo_coeff)

def get_mo_dip(bc, mol, origin=(0., 0., 0.)):
    with mol.with_common_origin(origin):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    return bc.rotate_focklike(ao_dip)

def make_vext_from_efield(efield, mo_dip):
    if efield is not None:
        v_ext = lambda t: -np.einsum('xij,x->ij', mo_dip, efield(t))
    else: 
        v_ext = lambda t: 0.0
    return v_ext

class RTTDSCF(lib.StreamObject):
    def __init__(self, mf, prop_method='magnus2', chkfile = None):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.max_memory = mf.max_memory
        self.prop = None
        self.trace = None
        self.chkfile = chkfile

        self.wfnsym = None
        if prop_method not in RTSCF_PROP_METHODS:
            raise ValueError(f'prop_method {prop_method} not recognized')
        self.prop_method = prop_method

    
    def kernel(self, t_end, dt, t_start=0.0, efield=None):
        bc = BasisChanger(self._scf.get_ovlp(), self._scf.mo_coeff)
        log = logger.new_logger(self, self.verbose)

        mo_dip = get_mo_dip(bc, self.mol)
        charges = self.mol.atom_charges()
        coords  = self.mol.atom_coords()
        nucl_dip = np.einsum('i,ix->x', charges, coords)

        v_ext = make_vext_from_efield(efield, mo_dip)
        
        self.trace = {'t': [], 'dipole': [], 'dm': []}

        if t_end <= t_start:
            raise ValueError('t_end must be greater than t_start')
        
        nsteps = math.ceil((t_end - t_start) / dt)

        chkf = h5py.File(self.chkfile, "w") if self.chkfile is not None else None
        if chkf is not None:
            chkf.create_dataset('t', (0,), maxshape=(None,), dtype=np.float64, chunks=True)
            chkf.create_dataset('dipole', (0, 3), maxshape=(None, 3), dtype=np.complex128, chunks=True)
            chkf.create_dataset('dm', (0, self.mol.nao, self.mol.nao),
                                dtype=np.complex128,
                                maxshape=(None, self.mol.nao, self.mol.nao),
                                chunks=(1, self.mol.nao, self.mol.nao))

        def stepcallback(t, dm):
            dipole = -np.sum(mo_dip * dm[None,:,:].real, axis=(1,2))
            self.trace['t'].append(t)
            self.trace['dipole'].append(dipole)
            self.trace['dm'].append(dm.copy())
            chkf['t'].resize((chkf['t'].shape[0] + 1), axis=0)
            chkf['dipole'].resize((chkf['dipole'].shape[0] + 1), axis=0)
            chkf['dm'].resize((chkf['dm'].shape[0] + 1), axis=0)
            chkf['t'][-1] = t
            chkf['dipole'][-1] = np.asarray(dipole, dtype=np.complex128)
            chkf['dm'][-1] = np.asarray(dm, dtype=np.complex128)

        

        if self.prop is None:
            if self.prop_method == 'magnus2':
                self.prop = magnus2.Magnus2Propagator(
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



        
        for _ in range(nsteps):
            self.prop.step()
