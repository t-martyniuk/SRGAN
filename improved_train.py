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

from models import Generator, Discriminator, FeatureExtractor, TVLoss


def calc_gradien_penalty(D, real, fake, use_cuda=False):
    minibatch = real.size()[0]
    alpha = torch.rand(minibatch, 1)
    alpha = alpha.expand(minibatch, real.nelement() / minibatch).contiguous().view(minibatch, 3, 32, 32)

    if use_cuda:
        alpha = alpha.cuda()

    interpolated = alpha * real.data + (1 - alpha) * fake.data
    interpolated = Variable(interpolated, requires_grad=True)

    if use_cuda:
        interpolated = interpolated.cuda()

    prob_interpolated = D(interpolated)

    gradients = grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda() if use_cuda else torch.ones(
                               prob_interpolated.size()),
                           create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(minibatch, -1)
    gradients_norm = gradients.norm(2, 1)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()
    return gradient_penalty


iterations = 100
iterations_d = 5
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
tv_loss = TVLoss()


if use_cuda:
    G.cuda()
    D.cuda()
    feature_extractor.cuda()


G_opt = optim.Adam(G.parameters(), lr=0.00001)
D_opt = optim.Adam(D.parameters(), lr=0.00001)


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
        D_real_loss = -D_real.mean()

        high_res_fake = G(low_res)
        D_fake = D(high_res_fake)
        D_fake_loss = D_fake.mean()

        gradient_penalty = calc_gradien_penalty(D, high_res_real, high_res_fake, use_cuda=use_cuda)

        D_loss = D_fake_loss - D_real_loss + gradient_penalty
        D_losses.append(D_loss.item())
        D_loss.backward()
        D_opt.step()

        if (i + 1) % iterations_d == 0:
            G.zero_grad()
            high_res_fake = G(low_res)

            real_features = Variable(feature_extractor(high_res_real).data)
            fake_features = feature_extractor(high_res_fake)

            G_loss_advers = 0.001 * (1 - D(high_res_fake)).mean()
            G_loss_content = content_criterion(high_res_fake, high_res_real)
            G_loss_features = 0.006 * content_criterion(fake_features, real_features)
            G_loss_tv_loss = 2e-8 * tv_loss(high_res_fake)
            G_loss = G_loss_advers  +  G_loss_content +  G_loss_features + G_loss_tv_loss
            G_losses.append(G_loss.item())
            G_loss.backward()
            G_opt.step()
            sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Advers/Content/Features): %.4f (%.4f/%.4f/%.4f)' %
                             ((iteration + 1), iterations, (i + 1), len(dataloader), D_loss.item(), G_loss.item(), G_loss_advers.item(), G_loss_content.item(), G_loss_features.item()))
    D_losses, G_losses = Tensor(D_losses), Tensor(G_losses)
    if use_cuda:
        D_losses, G_losses = D_losses.cuda(), G_losses.cuda()
    sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss: %.4f\n' % (
        (iteration + 1), iterations, (i + 1), len(dataloader), D_losses.mean().item(), G_losses.mean().item()))

            # Do checkpointing
    torch.save(G.state_dict(), str(models_path / 'generator_improved_cifar100.pth'))
    torch.save(D.state_dict(), str(models_path / 'discriminator_improved_cifar100.pth'))