
# Simple tool to decompress archive & convert npzs to npys

import os
from zipfile import ZipFile
import numpy as np

zipfiles = [x for x in os.listdir() if '.zip' in x]
npzfiles =[]
print('Uncompressing',len(zipfiles),' zip files.')
for zf in zipfiles:
    zfl = ZipFile(zf)
    for f in zfl.infolist():
        zfl.extract(f)
        npzfiles.append(f.filename)


# Replace decompressed npz files  with npys
print('Replacing ',len(npzfiles),' npz files with npy files.')
for ix,npzf in enumerate(npzfiles):
    with np.load(npzf,'r') as f:
        dat = f['arr_0']
    np.save(npzf.replace('npz','npy'),dat)
    os.remove(npzf)
    print('\rDone '+str(ix+1)+'/'+str(len(npzfiles)),end='')
