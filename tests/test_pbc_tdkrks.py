#!/usr/bin/env python

from pyscf import df
from pyscf.pbc import gto as pbcgto, scf as pbcscf, dft as pbcdft
import numpy as np
from rttddft.pbc.rttdbase import KRTTDSCF, kick_afield, gaussian_afield

def test_rttddft_diamond():

    '''
    TDSCF with k-point sampling
    '''
    cell = pbcgto.Cell()
    cell.atom = 'C 0 0 0; C 0.8925000000 0.8925000000 0.8925000000'
    cell.a = '''
    1.7850000000 1.7850000000 0.0000000000
    0.0000000000 1.7850000000 1.7850000000
    1.7850000000 0.0000000000 1.7850000000
    '''
    cell.pseudo = 'gth-hf-rev'
    cell.basis = {'C': [[0, (0.8, 1.0)],
                        [1, (1.0, 1.0)]]}
    cell.precision = 1e-10
    cell.build()
    kmesh = [2,1,1]
    kpts = cell.make_kpts(kmesh)
    mf = pbcdft.KRKS(cell, kpts=kpts, xc='pbe0').rs_density_fit(auxbasis=df.autoaux(cell)).run()

    step = 1.0
    afield = kick_afield(0.0, 0.0001, dir=(1.0,0.0,0.0))
    myrtd = KRTTDSCF(mf, prop_method='mmut')

    myrtd.kernel(2.0, step, afield=afield)
