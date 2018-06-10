import sys
from pathlib import Path

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.autograd import Variable, grad
from torchvision import datasets
from torchvision.transforms import transforms

from models import Generator, Discriminator, FeatureExtractor

iterations = 100
use_cuda = True
batch_size = 16
models_path = Path("models/")
models_path.mkdir(exist_ok=True, parents=True)

scale = transforms.Compose([transforms.ToPILImage(), transforms.Resize(16), transforms.ToTensor()])
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.CIFAR100(root="data/", train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


G = Generator(16, 2)
D = Discriminator()
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
content_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()

if use_cuda:
    G.cuda()
    D.cuda()
    feature_extractor.cuda()


G_opt = optim.Adam(G.parameters(), lr=0.00001)
D_opt = optim.Adam(D.parameters(), lr=0.00001)


ones_const = Variable(torch.ones(batch_size, 1))
zeros_const = Variable(torch.zeros(batch_size, 1))

if use_cuda:
    ones_const = ones_const.cuda()
    zeros_const = zeros_const.cuda()

for iteration in range(iterations):
    D_losses = []
    G_losses = []
    for i, (high_res_real, _) in enumerate(dataloader):
        minibatch = high_res_real.size(0)
        low_res = torch.FloatTensor(minibatch, 3, 16, 16)
        for j in range(minibatch):
            low_res[j] = scale(high_res_real[j])

        high_res_real = Variable(high_res_real)
        low_res = Variable(low_res)

        if use_cuda:
            high_res_real, low_res = high_res_real.cuda(), low_res.cuda()

        D.zero_grad()
        D_real = D(high_res_real)

        high_res_fake = G(low_res)
        D_fake = D(high_res_fake)

        D_loss = adversarial_criterion(D_fake, zeros_const) + adversarial_criterion(D_real, ones_const)
        D_losses.append(D_loss.item())
        D_loss.backward()
        D_opt.step()

        G.zero_grad()
        high_res_fake = G(low_res)

        real_features = Variable(feature_extractor(high_res_real).data)
        fake_features = feature_extractor(high_res_fake)

        G_loss_advers = adversarial_criterion(D(high_res_fake), ones_const)
        G_loss_content = content_criterion(high_res_fake, high_res_real)
        G_loss_features = content_criterion(fake_features, real_features)
        G_loss = 1e-3 * G_loss_advers  +  G_loss_content +  2e-6* G_loss_features
        G_losses.append(G_loss.item())
        G_loss.backward()
        G_opt.step()
        sys.stdout.write('\r[%d/%d][%d/%d] D_Loss: %.4f G_Loss (Adv/Cont/Feat): %.4f (%.4f/%.4f/%.4f)' %
                             ((iteration + 1), iterations, (i + 1), len(dataloader), D_loss.item(), G_loss.item(), G_loss_advers.item(), G_loss_content.item(), G_loss_features.item()))

    D_losses, G_losses = Tensor(D_losses), Tensor(G_losses)

    if use_cuda:
        D_losses, G_losses = D_losses.cuda(), G_losses.cuda()
    sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss: %.4f\n' % (
        (iteration + 1), iterations, (i + 1), len(dataloader), D_losses.mean().item(), G_losses.mean().item()))

            # Do checkpointing
    torch.save(G.state_dict(), str(models_path / 'generator_cifar100.pth'))
    torch.save(D.state_dict(), str(models_path / 'discriminator_cifar100.pth'))