import h5py
import numpy as np
from models.bert_encoder import get_bert_txt_embeddings

dataset = h5py.File('/w/247/zyanliu/csc2516_project/Text-to-Image-Synthesis/birds_clip.hdf5', mode='r')
print(dataset.keys())

new_dataset = h5py.File('birds_bert.hdf5','w')
for split in dataset.keys():
    # train/valid/test
    split_dataset = new_dataset.create_group(split)
    dataset_keys = [str(k) for k in dataset[split].keys()]
    for example_name in dataset_keys:

        example = dataset[split][example_name]
        # print(np.array(example['txt']))
        # print(example['embeddings'].shape)
        txt = np.array(example[b'txt']).astype('U')
        embedding = get_bert_txt_embeddings(txt.tolist())[0]
        # print(embedding.shape, '-------')
        ex = split_dataset.create_group(example_name)
        ex.create_dataset('name', data=example['name'])
        ex.create_dataset('img', data=example['img'])
        ex.create_dataset('embeddings', data=embedding) # the embedding of given txt, shape (1024)
        ex.create_dataset('class', data=example['class'])
        ex.create_dataset('txt', data=example[b'txt'])


