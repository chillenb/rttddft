import numpy as np               
import os

from pyscf.pbc import df, gto, dft, scf
from pyscf import lib

cell = gto.Cell()
cell.build(
    a=np.eye(3) * 3.0,
    atom='He 0 0 0',
    basis = 'gth-dzvp',
    pseudo = 'gth-pade',
    dimension=3,
    max_memory=1000,
    verbose=6,
    precision=1e-14,
)                     

kpts = cell.make_kpts([2, 2, 2])
gdf = df.GDF(cell, kpts)
gdf_fname = 'he3_gdf.h5'
gdf._cderi_to_save = gdf_fname
                                                           
if not os.path.isfile(gdf_fname):
    gdf.build()

mf = scf.KRHF(cell, kpts).density_fit()
mf.with_df = gdf
mf.chkfile = 'he3.chk'
mf.exxdiv = None
mf.kernel()

