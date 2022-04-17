# Text-to-Image-Synthesis with CLIP/Bert Embedding

## Intoduction

This is a project of csc-2516. Based on the pytorch implementation of [Generative Adversarial Text-to-Image Synthesis paper](https://arxiv.org/abs/1605.05396), we train a conditional generative adversarial network, conditioned on text descriptions, to generate images that correspond to the description. The network architecture is shown below (Image from [1]). This architecture is based on DCGAN. The text encoder can be either CLIP Text Encoder or BERT.

<figure><img src='images/pipeline.png'></figure>
Image credits [1]

## Requirements

- pytorch 
- visdom
- h5py == 2.10.1
- PIL
- numpy

use `t2i_env.yaml` to create your environment
## Datasets

We used [Caltech-UCSD Birds 200](http://www.vision.caltech.edu/visipedia/CUB-200.html) datasets, we converted each dataset (images, text embeddings) to hd5 format. 

We used the [text embeddings](https://github.com/reedscot/icml2016) provided by the paper authors

**To use this code you can either:**

- Use the converted hd5 datasets,  [birds_bert](https://drive.google.com/file/d/1wVeY_W_d0f5QAuSkVf4GaY_m16XPNOoI/view?usp=sharing), [birds_clip](https://drive.google.com/file/d/1c1-cmBqllLbXVeCSY_7MSt8D9Agh4_Yg/view?usp=sharing), [birds_origin]()
- Convert the data youself
  1. download the dataset as described [here](https://github.com/reedscot/cvpr2016)
  2. Add the paths to the dataset to `config_[xxx].yaml` file.
  3. Use the scripts under [convert_dataset](convert_dataset) to convert the dataset.
  
**Hd5 file taxonomy**
`
 - split (train | valid | test )
    - example_name
      - 'name'
      - 'img'
      - 'embeddings'
      - 'class'
      - 'txt'
      
## Usage
### Training

`python [runtime.py|runtime_clip.py|runtime_bert.py]`

**Arguments:**
- `type` : GAN archiecture to use `(gan | wgan | vanilla_gan | vanilla_wgan)`. default = `gan`. Vanilla mean not conditional
- `dataset`: Dataset to use `(birds | flowers)`. default = `flowers`
- `split` : An integer indicating which split to use `(0 : train | 1: valid | 2: test)`. default = `0`
- `lr` : The learning rate. default = `0.0002`
- `diter` :  Only for WGAN, number of iteration for discriminator for each iteration of the generator. default = `5`
- `vis_screen` : The visdom env name for visualization. default = `gan`
- `save_path` : Path for saving the models.
- `l1_coef` : L1 loss coefficient in the generator loss fucntion for gan and vanilla_gan. default=`50`
- `l2_coef` : Feature matching coefficient in the generator loss fucntion for gan and vanilla_gan. default=`100`
- `pre_trained_disc` : Discriminator pre-tranined model path used for intializing training.
- `pre_trained_gen` Generator pre-tranined model path used for intializing training.
- `batch_size`: Batch size. default= `64`
- `num_workers`: Number of dataloader workers used for fetching data. default = `8`
- `epochs` : Number of training epochs. default=`200`
- `cls`: Boolean flag to whether train with cls algorithms or not. default=`False`
- `embed_dim`： text encoder embedding dimension, e.g., `512`
- `config_path`:  config_path, e.g., `config_clip.yaml`
- `checkpoints_path`, path to save checkpoints, e.g., `checkpoints_clip`


**Before training:**
- Remember to install [visdom](https://github.com/fossasia/visdom) and start the server.

**Inference**

- See the script [script_inference_clip.sh|script_inference.sh] to get the qualitative results from test dataset.

