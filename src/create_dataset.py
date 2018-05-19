
import PIL
import PIL.Image
import PIL.ImageChops
import PIL.ImageDraw
import PIL.ImageFont
import PIL.ImageOps
import h5py
import numpy
import glob
import json
from tqdm import tqdm
from typing import List


def create_dataset(dataset_name: str, font_list: List[str]):
    w, h = 75, 75
    chars = [''.join("uni" + str(hex(hexa)[2:].upper())) for hexa in range(0xAC00, 0xD7A4)]
    dataset_path = f'{dataset_name}.hdf5'

    f = h5py.File(f'{dataset_name}.hdf5', 'w')
    dataset = f.create_dataset(dataset_name, shape=(1, len(chars), h, w), chunks=(1, len(chars), h, w), maxshape=(None, len(chars), h, w), dtype='u1')

    def read_font(fn):
        data = []
        imgs = list(glob.glob('fonts/' + fn + '_*.png'))
        imgs.sort()

        for img_name in tqdm(imgs):
            img = PIL.Image.open(img_name)
            matrix = numpy.array(img.getdata(1)).reshape((h, w))  # get greyscale ch:1
            # matrix = 255 - matrix  # TODO: WHY!!!!!!!!!!!!
            data.append(matrix)

        return numpy.array(data)

    i = 0
    for font in font_list:
        print(font)
        try:
            data = read_font(font)
        except Exception as e:  # IOError:
            print('was not able to read', font)
            print(e)
            continue

        print(data.shape)
        dataset.resize((i + 1, len(chars), h, w))  # TODO: WHY!!!!!
        dataset[i] = data
        i += 1
        f.flush()
    f.close()

    dataset_manifest = json.dumps({'_type': 'dataset_manifest',
                                   '_version': 1,
                                   'name': dataset_name,
                                   'path': dataset_path,
                                   'fonts': font_list,
                                   'w': w,
                                   'h': h
                                   })
    with open(f'{dataset_name}.json', 'w') as f:
        f.write(dataset_manifest)
