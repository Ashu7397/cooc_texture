import argparse
import datetime
import os
import random

import torch
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()

parser.add_argument('--texturePath', default='.', help='path to texture image')
parser.add_argument('--modelPath', default='.', help='path to saved model folder')
parser.add_argument('--testModel', type=bool, default=False, help='use the model for testing')
parser.add_argument('--evalFunc', default='.', help='function of evaluation')
parser.add_argument('--checkpointNumber', type=int, default=120, help='checkpoint number for model')
parser.add_argument('--outputFolder', default='.', help='folder to output images and model checkpoints')

parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--upscaleSize', type=int, default=1200, help='the height / width of the upscaled image of network')
parser.add_argument('--ngf', type=int, default=120,
                    help='number of channels of generator (at largest spatial resolution)')
parser.add_argument('--ndf', type=int, default=120,
                    help='number of channels of discriminator (at largest spatial resolution)')
parser.add_argument('--WGAN', type=bool, default=True, help='use WGAN-GP adversarial loss')
parser.add_argument('--LS', type=bool, default=False, help='use least squares GAN adversarial loss')

parser.add_argument('--niter', type=int, default=120, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--dIter', type=int, default=1, help='number of Discriminator steps -- for 1 Generator step')

parser.add_argument('--addNoise', type=bool, default=True, help='Add Noise dimension to cooc')
parser.add_argument('--coocNorm', type=bool, default=True, help='Apply uniform normalization b/w (-1, 1) to cooc too')
parser.add_argument('--noiseFirst', type=bool, default=True, help='[D] Concat Noise first and then Cooc')
parser.add_argument('--coocLoss', type=bool, default=True, help='[D] Put condition on generator over Cooc Loss')
parser.add_argument('--useKLCooc', type=bool, default=False, help='[D] Use KL Divergence loss for Cooc condition')
parser.add_argument('--useL1Cooc', type=bool, default=True, help='[D] Use L1 loss for Cooc condition')

parser.add_argument('--useSavedWeights', default='', help='Use weight from earlier simulation. Give path to weights')
parser.add_argument('--kVal', type=int, default=8, help='Number of clusters for kMeans')
parser.add_argument('--noConcat', type=bool, default=False, help='Discriminator implicit condition')
parser.add_argument('--useVolLoss', type=bool, default=False, help='Use entire vol for the explicit loss')
parser.add_argument('--lambdaVal', type=int, default=1, help='Value of Gradient Penalty')
opt = parser.parse_args()

nDep = 5  # opt.nDep  #Depth of out network!

##GAN criteria changes given loss options LS or WGAN
if not opt.WGAN and not opt.LS:
    criterion = torch.nn.BCELoss()
elif opt.LS:
    def crit(x, l):
        return ((x - l) ** 2).mean()


    criterion = crit
else:
    def dummy(val, label):
        return (val * (1 - 2 * label)).mean()  # so -1 fpr real. 1 fpr fake


    criterion = dummy

if opt.outputFolder == '.':
    # TODO: Add check for testing function
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if os.path.isfile(opt.texturePath):
        img_name = opt.texturePath.split(os.path.sep)[-1].split('.')[0]
        opt.outputFolder = "results/" + img_name + '_'
    else:
        i = opt.texturePath[:-1].rfind('/')
        opt.outputFolder = "results/" + opt.texturePath[i + 1:]
    opt.outputFolder += stamp + "/"
try:
    os.makedirs(opt.outputFolder)
except OSError:
    pass
print("outputFolderolder: " + opt.outputFolder)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 999999)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

text_file = open(opt.outputFolder + "options.txt", "w")
text_file.write(str(opt))
text_file.write('\nUsed Random Seed!')
text_file.close()
print(opt)

canonicT = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
if opt.coocNorm:
    canonicT_cooc = [transforms.ToTensor(), transforms.Normalize(torch.zeros(opt.kVal ** 2), torch.ones(opt.kVal ** 2))]
else:
    canonicT_cooc = [transforms.ToTensor()]
TRANSFORM_TEX = transforms.Compose(canonicT)
TRANSFORM_TEX_COOC = transforms.Compose(canonicT_cooc)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE", DEVICE)
