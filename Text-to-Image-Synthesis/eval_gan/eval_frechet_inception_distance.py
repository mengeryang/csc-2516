# import sys
# sys.path.append("c:\\Users\\micha\\Desktop\\csc-2516\\Text-to-Image-Synthesis")

import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import numpy as np
from scipy import linalg
from tqdm import tqdm
from scipy.stats import entropy

from eval_gan.inception_dataset import InceptionDataset
from eval_gan.train_inception_net import InceptionV3
from models.gan_factory_clip import gan_factory

def frechet_inception_distance(args, cuda=True, resize=False):
    """Computes the frechet inception distance of the generated images
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    """

    imgs = InceptionDataset(args.dataset_type, args.dataset_path, split=args.split)
    N = len(imgs)

    batch_size = args.batch_size

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size, num_workers=args.num_workers, drop_last=True)

    # Load inception model
    inception_model = InceptionV3(args.n_class).type(dtype)
    state_dict = torch.load(args.model_path)
    inception_model.load_state_dict(state_dict)
    inception_model.train()

    # load generator
    generator = torch.nn.DataParallel(gan_factory.generator_factory(args.type, args.embed_dim))
    generator.load_state_dict(torch.load(args.pre_trained_gen))

    up = nn.Upsample(size=(299, 299), mode='bicubic', align_corners=True).type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        with torch.no_grad():
            feat, x = inception_model(x)
        return feat.cpu().numpy()

    # Get predictions
    features_fake = np.zeros((N, 2048))
    features_real = np.zeros((N, 2048))

    for i, batch in enumerate(tqdm(dataloader, mininterval=args.tqdm_interval), 0):
        right_images = batch['right_images'].float().type(dtype)
        right_images = Variable(right_images)
        batch_size_i = right_images.size()[0]

        right_embed = batch['right_embed']
        right_embed = Variable(right_embed.float()).type(dtype)
        

        noise = Variable(torch.randn(right_images.size(0), 100)).type(dtype)
        noise = noise.view(noise.size(0), 100, 1, 1)
        fake_images = generator(right_embed, noise)

        features_fake[i*batch_size:i*batch_size + batch_size_i] = get_pred(fake_images)

        features_real[i*batch_size:i*batch_size + batch_size_i] = get_pred(right_images)

    mu_fake = np.mean(features_fake, axis=0)
    sigma_fake = np.cov(features_fake, rowvar=False)

    mu_real = np.mean(features_real, axis=0)
    sigma_real = np.cov(features_real, rowvar=False)

    fid_value = calculate_frechet_distance(mu_fake, sigma_fake, mu_real, sigma_real)

    return fid_value

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument("--model_path", default='./dataset/inception_v3_latest_flowers.pth')
    parser.add_argument('--dataset_type', default='flowers', choices=['birds', 'flowers'], type=str)
    parser.add_argument('--dataset_path', default='./dataset/flowers_clip.hdf5')
    parser.add_argument('--dataset_split', default='test', type=str, help="separate by comma")
    parser.add_argument('--n_class', default=102, type=int)
    parser.add_argument('--print_interval', default=5, type=int)
    parser.add_argument('--tqdm_interval', default=60, type=float)

    # gan
    parser.add_argument("--type", default='gan')
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--pre_trained_gen', default="./checkpoints_clip/flowers/clip_gan_cls/gen_190.pth")
    args = parser.parse_args()

    resize = True

    args.split = [split for split in args.dataset_split.split(",")]

    print ("Calculating Frechet Inception Distance...")
    # only support cuda=True
    print (frechet_inception_distance(args, cuda=True, resize=resize))

    # ==================== birds result ====================
    # original

    # bert_gan
    # bert_gan_cls
    # bert_gan_cls_int

    # clip_gan
    # clip_gan_cls
    # clip_gan_cls_int

    # ==================== flowers result ====================
    # 2.750164230728835 bert_gan_cls
    # 2.5694797126851867 clip_gan_cls
