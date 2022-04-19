# import sys
# sys.path.append("c:\\Users\\micha\\Desktop\\csc-2516\\Text-to-Image-Synthesis")

import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy

from inception_score.inception_dataset import InceptionDataset
from inception_score.train_inception_net import InceptionV3
from models.gan_factory_clip import gan_factory

def inception_score(args, cuda=True, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """

    imgs = InceptionDataset(args.dataset_path, split=['test'])
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
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size, num_workers=args.num_workers)

    # Load inception model
    inception_model = InceptionV3(200).type(dtype)
    state_dict = torch.load(args.model_path)
    inception_model.load_state_dict(state_dict)
    inception_model.train()

    # load generator
    if args.mode == "fake":
        generator = torch.nn.DataParallel(gan_factory.generator_factory(args.type, args.embed_dim))
        generator.load_state_dict(torch.load(args.pre_trained_gen))

    up = nn.Upsample(size=(299, 299), mode='bicubic', align_corners=True).type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        with torch.no_grad():
            x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 200))

    for i, batch in enumerate(tqdm(dataloader, mininterval=args.tqdm_interval), 0):
        right_images = batch['right_images'].float().type(dtype)
        right_images = Variable(right_images)
        batch_size_i = right_images.size()[0]
        if args.mode == "fake":
            right_embed = batch['right_embed']
            right_embed = Variable(right_embed.float()).type(dtype)
        

            noise = Variable(torch.randn(right_images.size(0), 100)).type(dtype)
            noise = noise.view(noise.size(0), 100, 1, 1)
            fake_images = generator(right_embed, noise)

            preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(fake_images)
        else:
            preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(right_images)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument("--model_path", default='./dataset/inception_v3_latest.pth')
    parser.add_argument('--dataset_path', default='./dataset/birds_clip.hdf5')
    parser.add_argument('--print_interval', default=5, type=int)
    parser.add_argument('--tqdm_interval', default=60, type=float)
    parser.add_argument('--mode', default="fake", type=str)

    # gan
    parser.add_argument("--type", default='gan')
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--pre_trained_gen', default="./checkpoints_clip/gen_190.pth")
    args = parser.parse_args()

    resize = True if args.mode == "fake" else False

    print ("Calculating Inception Score...")
    # only support cuda=True
    print (inception_score(args, cuda=True, resize=resize, splits=10))
    # inception score for real test dataset: (21.93618796142134, 0.8870024899660087)
    # (64.44085900313355, 1.2005505343822582) original
    # (63.14714825591868, 1.4279450332975985) bert
    # (65.95682410638216, 0.9856238497738542) clip