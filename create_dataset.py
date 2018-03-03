import sys

import PIL
import PIL.Image
import PIL.ImageChops
import PIL.ImageDraw
import PIL.ImageFont
import PIL.ImageOps
import h5py
import numpy
import keras
from sklearn.preprocessing import OneHotEncoder

w, h = 75, 75
chars = [''.join("uni" + str(hex(hexa)[2:].upper())) for hexa in range(0xAC00, 0xD7A4)]


import glob
def read_font(fn):
    data = []
    imgs = list(glob.glob(fn + '_*.png'))
    imgs.sort()

    for img_name in imgs:
        img_sq = PIL.Image.open(img_name)
        matrix = numpy.array(img_sq.getdata()).reshape((h, w))
        # matrix = 255 - matrix  # TODO: WHY!!!!!!!!!!!!
        data.append(matrix)

    return numpy.array(data)


from tqdm import tqdm

                
f = h5py.File('fonts.hdf5', 'w')
dset = f.create_dataset('fonts', (1, len(chars), h, w), chunks=(1, len(chars), h, w), maxshape=(None, len(chars), h, w), dtype='u1')

i = 0
for fn in tqdm(sys.argv[1:]):
    print(fn)
    try:
        data = read_font(fn)
    except: # IOError:
        print('was not able to read', fn)
        continue

    print(data.shape)
    dset.resize((i+1, len(chars), h, w))  # TODO: WHY!!!!!
    dset[i] = data
    i += 1
    f.flush()

f.close()
