import h5py
import numpy as np

split = 'test'
# split = 'train'

dataset = h5py.File('/w/247/zyanliu/csc2516_project/Text-to-Image-Synthesis/birds_origin.hdf5', mode='r')
print(dataset.keys())
dataset_keys = [str(k) for k in dataset[split].keys()]

print(len(dataset_keys))
for example_name in dataset_keys:
    example = dataset[split][example_name]
    try:
        txt = np.array(example[b'txt']).astype('U') # revise astype(str)
    except:
        print(example_name, 'wrong')
        assert(1==2)