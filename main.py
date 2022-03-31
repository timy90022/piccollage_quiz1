import os
import ipdb

from model import *
from datasets import *

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter


def main():
    hyper_param_epoch = 60
    hyper_param_batch = 2048
    hyper_param_learning_rate = 0.0002

    transforms_train = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
    
    transforms_test = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])

    data_path = '/home/timy90022/collage/correlation_assignment/'

    train_data_set = CustomImageDataset(data_set_path=data_path, transforms=transforms_train, type_='train')
    train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch, shuffle=True, num_workers=14)

    test_data_set = CustomImageDataset(data_set_path=data_path, transforms=transforms_test, type_='test')
    test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=True, num_workers=4)

    # tensorboardX
    writer = SummaryWriter()

    # Network
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    custom_model = ResNet18(1).to(device)

    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(custom_model.parameters(), lr=hyper_param_learning_rate)

    for e in range(hyper_param_epoch):
        custom_model.train()
        for n_iter, item in enumerate(train_loader):
            images = item['image'].to(device)
            labels = item['label'].to(device)

            # Forward pass
            outputs = custom_model(images).reshape(-1)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if n_iter == 50:
            #     ipdb.set_trace()
            writer.add_scalar('data/scalar1', loss.item(), n_iter+e*len(train_loader))

            if (n_iter + 1) % 10 == 1:
                print('Epoch [{}/{}], Loss: {:.4f}'
                    .format(e + 1, hyper_param_epoch, loss.item()))
        
        # Update learning rate
        for g in optimizer.param_groups:
            g['lr'] *= 0.9

        # Test the model
        custom_model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct_s = 0
            correct_m = 0
            correct_l = 0
            total = 0
            for item in test_loader:
                images = item['image'].to(device)
                labels = item['label'].to(device)
                outputs = custom_model(images).reshape(-1)
                diff = torch.absolute(outputs - labels)
                loss = criterion(outputs, labels)

                correct_s += torch.sum(torch.le(diff, 0.01))
                correct_m += torch.sum(torch.le(diff, 0.1))
                correct_l += torch.sum(torch.le(diff, 0.3))
                total += len(labels)


            print('Test Accuracy of the model on the {} test images: {:.3f} %, loss: {:.3f}'.format(total, 100 * correct_l / total, loss))
            writer.add_scalar('data/Acc(0.01)', correct_s / total, e)
            writer.add_scalar('data/Acc(0.1)', correct_m / total, e)
            writer.add_scalar('data/Acc(0.3)', correct_l / total, e)

    writer.close()

if __name__ == '__main__':
    main()