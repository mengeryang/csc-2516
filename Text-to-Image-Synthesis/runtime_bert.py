from trainer_clip import Trainer
import argparse
from PIL import Image
import os

parser = argparse.ArgumentParser()
parser.add_argument("--type", default='gan')
parser.add_argument("--lr", default=0.0002, type=float)
parser.add_argument("--l1_coef", default=50, type=float)
parser.add_argument("--l2_coef", default=100, type=float)
parser.add_argument("--diter", default=5, type=int)
parser.add_argument("--cls", default=False, action='store_true')
parser.add_argument("--disable_visdom", default=False, action='store_true')
parser.add_argument("--vis_screen", default='gan')
parser.add_argument("--save_path", default='')
parser.add_argument("--inference", default=False, action='store_true')
parser.add_argument('--pre_trained_disc', default=None)
parser.add_argument('--pre_trained_gen', default=None)
parser.add_argument('--dataset', default='birds')
parser.add_argument('--split', default=0, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=6, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--embed_dim', default=768, type=int)
parser.add_argument('--config_path', default='config_bert.yaml', type=str)
parser.add_argument('--checkpoints_path', default='checkpoints_bert', type=str)
args = parser.parse_args()

trainer = Trainer(type=args.type,
                  dataset=args.dataset,
                  split=args.split,
                  lr=args.lr,
                  diter=args.diter,
                  vis_screen=args.vis_screen,
                  save_path=args.save_path,
                  l1_coef=args.l1_coef,
                  l2_coef=args.l2_coef,
                  pre_trained_disc=args.pre_trained_disc,
                  pre_trained_gen=args.pre_trained_gen,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers,
                  epochs=args.epochs,
                  config_path = args.config_path,
                  embed_dim = args.embed_dim,
                  checkpoints_path = args.checkpoints_path,
                  disable_visdom = args.disable_visdom
                  )

if not args.inference:
    trainer.train(args.cls)
else:
    trainer.predict()

