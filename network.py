import torch
import torch.nn as nn
from config import opt
from torch import autograd

norma = nn.BatchNorm2d

def calc_gradient_penalty(netD, cooc_vol_D, real_data, fake_data):
    LAMBDA=opt.lambdaVal
    BATCH_SIZE=fake_data.shape[0]
    alpha = torch.rand(BATCH_SIZE).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    device=real_data.get_device()
    alpha = alpha.to(device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    disc_interpolates = netD(interpolates, cooc_vol_D)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# '''
class Discriminator(nn.Module):
    # @param ncIn is input channels
    # @param ndf is channels of first layer, doubled up after every conv. layer with stride
    # @param nDep is depth
    # @param bSigm is final nonlinearity, off for some losses
    def __init__(self, ndf, nDep, ncIn=3, cooc_size=-1, bSigm=True):
        super(Discriminator, self).__init__()
        assert cooc_size > 0
        layers = []
        of = ncIn
        for i in range(nDep):
            if i==nDep-1:
                nf=1
            elif i == nDep - 2 and not opt.noConcat:
                of = of + cooc_size
                nf = ndf * 2 ** i
            else:
                nf = ndf * 2 ** i
            layers+=[nn.Conv2d(of, nf, 5, 2, 2)]
            if i !=0 and i !=nDep-1:
                if not opt.noConcat:
                    bn_nf = nf + cooc_size if i == nDep - 3 else nf
                if not opt.WGAN:
                    layers+=[norma(bn_nf)]

            if i < nDep -1:
                layers+=[nn.LeakyReLU(0.2, inplace=True)]
            else:
                if bSigm:
                    layers+=[nn.Sigmoid()]
            of = nf
        if opt.noConcat:
            self.main = nn.Sequential(*layers)
            self.after_concat = None
        else:
            self.main = nn.Sequential(*layers[0:6])
            self.after_concat = nn.Sequential(*layers[6:])

    def forward(self, input, cooc_input=None):
        main_output = self.main(input)
        if cooc_input is not None:
            main_output = torch.cat((main_output, cooc_input), dim=1)
        if not opt.noConcat:
            output = self.after_concat(main_output)
        if opt.WGAN:
            return output.mean(3).mean(2).unsqueeze(2).unsqueeze(3)
        return output


##################################################
class NetG(nn.Module):
    # @param ngf is channels of first layer, doubled up after every stride operation, or halved after upsampling
    # @param nDep is depth, both of decoder and of encoder
    # @param nz is dimensionality of stochastic noise we add
    def __init__(self, ngf, nDep, nz, nc=3):
        super(NetG, self).__init__()

        of = nz
        layers = []
        for i in range(nDep):

            if i == nDep - 1:
                nf = nc
            else:
                nf = ngf * 2 ** (nDep - 2 - i)

            layers += [nn.Upsample(scale_factor=2, mode='nearest')]  # nearest is default anyway
            layers += [nn.Conv2d(of, nf, 5, 1, 2)]
            if i == nDep - 1:
                layers += [nn.Tanh()]
            else:
                if opt.WGAN:
                    layers += [norma(nf)]
                layers += [nn.ReLU(True)]
            of = nf
        self.G = nn.Sequential(*layers)

    def forward(self, input):
        return self.G(input)