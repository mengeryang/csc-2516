import argparse
from tqdm import tqdm
import numpy as np
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
    model = InceptionV3(args.n_class).cuda().train()
    dataset = InceptionDataset(args.dataset_type, args.dataset_path, split=args.split)
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
            print("")
            print(f"Epoch {epoch + 1} training loss: {running_loss / iterations}")
            print("")
        
        if (epoch + 1) % 10 == 0:
            print("")
            print("Compute training accuracy...")
            right = 0.
            total = 0.
            for sample in tqdm(dataloader, mininterval=args.tqdm_interval):
                imgs = sample['right_images'].float().cuda()
                labels = sample['right_classes'].squeeze(1).numpy()

                with torch.no_grad():
                    logit = model(imgs)
                    prob, class_ = torch.max(nn.functional.softmax(logit, dim=1), dim=1)

                class_ = class_.cpu().numpy()
                right += np.sum(class_ == labels)
                total += len(class_)
            accuracy = right / total
            print("")
            print(f"Training accuracy: {accuracy}")
            print("")
            torch.save(model.state_dict(), args.save_path + f"inception_v3_{epoch + 1}_{args.dataset_type}.pth")

        if (epoch + 1) % 2 == 0:
            scheduler.step()

    torch.save(model.state_dict(), args.save_path + f"inception_v3_latest_{args.dataset_type}.pth")
    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.005, type=float)
    parser.add_argument("--gamma", default=0.94, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument("--save_path", default='/scratch/gobi2/wren/2516/', type=str)
    parser.add_argument('--dataset_type', default='birds', choices=['birds', 'flowers'], type=str)
    parser.add_argument('--dataset_path', default='./dataset/birds.hdf5', type=str)
    parser.add_argument('--dataset_split', default='train,valid,test', type=str, help="separate by comma")
    parser.add_argument('--print_interval', default=5, type=int)
    parser.add_argument('--tqdm_interval', default=60, type=float)
    args = parser.parse_args()

    args.split = [split for split in args.dataset_split.split(",")]

    class_dict = {"train":150, "valid":50, "test":50} if args.dataset_type == "birds" else {"train":62, "valid":20, "test":20}

    n_class = 0
    for split in args.split:
        n_class += class_dict[split]
    args.n_class = n_class

    train(args)