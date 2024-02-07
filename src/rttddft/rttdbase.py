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

class RTTDSCF(lib.StreamObject):
    def __init__(self, mf):
        self.mol = mf.mol
        self._scf = mf
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.max_memory = mf.max_memory
        self.chkfile = mf.chkfile

        self.wfnsym = None