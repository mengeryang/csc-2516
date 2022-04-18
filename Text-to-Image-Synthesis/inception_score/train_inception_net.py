import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from inception_score.inception_dataset import InceptionDataset


class InceptionV3(nn.Module):
    def __init__(self, n_class=200):
        super(InceptionV3, self).__init__()

        self.inception = inception_v3(pretrained=True)
        self.inception.fc = nn.Identity()

        self.fc = nn.Linear(2048, n_class)

    def forward(self, x):
        x = self.inception(x).logits
        return self.fc(x)


def train(args):
    model = InceptionV3().cuda().train()
    dataset = InceptionDataset(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)

    for epoch in range(args.epochs):
        running_loss = 0
        iterations = 0
        for sample in tqdm(dataloader, mininterval=args.tqdm_interval):
            imgs = sample['right_images'].float().cuda()
            labels = sample['right_classes'].squeeze(1).cuda()

            logit = model(imgs)
            loss = criterion(logit, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            iterations += 1

        if (epoch + 1) % args.print_interval == 0:
            print(f"Epoch {epoch + 1} training loss: {running_loss / iterations}")

        if (epoch + 1) % 2 == 0:
            scheduler.step()

    torch.save(model.state_dict(), args.save_path)
    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.05, type=float)
    parser.add_argument("--gamma", default=0.94, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument("--save_path", default='/scratch/gobi2/wren/2516/inception_v3.pth')
    parser.add_argument('--dataset_path', default='./dataset/birds.hdf5')
    parser.add_argument('--print_interval', default=5, type=int)
    parser.add_argument('--tqdm_interval', default=60, type=float)
    args = parser.parse_args()

    train(args)