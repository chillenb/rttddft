#!/usr/bin/env python

from pyscf import df
from pyscf.pbc import gto as pbcgto, scf as pbcscf, dft as pbcdft
import numpy as np
import pytest
from rttddft.pbc.rttdbase import KRTTDSCF, kick_afield, gaussian_afield

@pytest.mark.parametrize("prop_method", ['magnus2', 'mmut'])
def test_rttddft_diamond(prop_method):
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
    myrtd = KRTTDSCF(mf, prop_method=prop_method)

    myrtd.kernel(2.0, step, afield=afield)



@pytest.mark.parametrize("prop_method", ['magnus2', 'mmut'])
def test_rttddft_diamond_ao_mo(prop_method):
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
    myrtd_ao = KRTTDSCF(mf, prop_method=prop_method)
    myrtd_ao.kernel(2.0, step, afield=afield, mo_basis=False)

    myrtd_mo = KRTTDSCF(mf, prop_method=prop_method)
    myrtd_mo.kernel(2.0, step, afield=afield, mo_basis=True)

    C = mf.mo_coeff
    for t1, dip1, t2, dip2 in zip(myrtd_mo.trace['t'], myrtd_mo.trace['dipole'],
                          myrtd_ao.trace['t'], myrtd_ao.trace['dipole']):
        assert abs(t1 - t2) < 1e-8
        assert np.allclose(dip1, dip2, atol=1e-6)

    for t1, dm_mo, t2, dm_ao in zip(myrtd_mo.trace['t'], myrtd_mo.trace['dm'],
                          myrtd_ao.trace['t'], myrtd_ao.trace['dm']):
        assert abs(t1 - t2) < 1e-8
        for k in range(len(kpts)):
            assert np.allclose(C[k] @ dm_mo[k] @ C[k].conj().T, dm_ao[k], atol=1e-6)
