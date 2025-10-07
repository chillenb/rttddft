from pyscf import gto, dft
from rttddft.rttdbase import RTTDSCF, gpulse_efield, kick_field


def test_rttddft_water():

    mol = gto.Mole()
    mol.build(
        atom = '''
        O     0.00000000    -0.00001441    -0.34824012
        H    -0.00000000     0.76001092    -0.93285191
        H     0.00000000    -0.75999650    -0.93290797
        ''',
        basis = '6-31G',
        symmetry = True,
        verbose=5,
    )

    mf = dft.RKS(mol)
    mf.xc = 'pbe0'
    mf.kernel()


    step = 0.4
    efield = kick_field(step/2, 0.0001, dir=(0,0,1.0))
    myrtd = RTTDSCF(mf)

    myrtd.kernel(4.0, step, efield=efield)
