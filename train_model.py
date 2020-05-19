import math
import sys
import time

import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from tqdm import tqdm

import cooc_utils
from config import opt, nDep, criterion, TRANSFORM_TEX, TRANSFORM_TEX_COOC
from network import weights_init, Discriminator, calc_gradient_penalty, NetG
from utils import TextureDataset, setNoise, setCooc


def get_loss_for_cooc(fake_im, cooc_vol_main, cooc_matrix):
    if opt.useVolLoss:
        cooc_fake_im = cooc_utils.cooc_finder_tensor(fake_im, dataset.im_gmm, dataset.f_cooc, dataset.k_clusters,
                                                     max_batch=4000, in_network=True)
        if opt.useKLCooc:
            loss_cooc = cooc_kl_criterion(cooc_fake_im, cooc_vol_main.squeeze(1))
        elif opt.useL1Cooc:
            loss_cooc = cooc_l1_criterion(cooc_fake_im, cooc_vol_main.squeeze(1))
        else:
            loss_cooc = cooc_l2_criterion(cooc_fake_im, cooc_vol_main.squeeze(1))
    else:
        cooc_fake_im, _ = cooc_utils.collect_cooc_batch(fake_im, dataset.im_gmm, dataset.f_cooc)
        if opt.useKLCooc:
            loss_cooc = cooc_kl_criterion(cooc_fake_im, cooc_matrix.squeeze(1))
        elif opt.useL1Cooc:
            loss_cooc = cooc_l1_criterion(cooc_fake_im, cooc_matrix.squeeze(1))
        else:
            loss_cooc = cooc_l2_criterion(cooc_fake_im, cooc_matrix.squeeze(1))
    return loss_cooc


cudnn.benchmark = True

img_list = []
G_losses = []
D_losses = []

cooc_size = opt.kVal * opt.kVal
if opt.addNoise:
    total_z = cooc_size * 2
    noise_dim = cooc_size
    assert noise_dim > 0
else:
    total_z = cooc_size

noise_spatial_dim = opt.imageSize // 2 ** nDep
factor_scale = noise_spatial_dim / opt.imageSize

dataset = TextureDataset(opt.texturePath, factor_scale, TRANSFORM_TEX, TRANSFORM_TEX_COOC)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)

ngf = int(opt.ngf)
ndf = int(opt.ndf)

desc = "fc" + '1.0' + "_ngf" + str(ngf) + "_ndf" + str(ndf) + "_dep" + str(nDep) + "-" + str(nDep)

if opt.WGAN:
    desc += '_WGAN'
if opt.LS:
    desc += '_LS'
netD = Discriminator(ndf, nDep, cooc_size=cooc_size, bSigm=not opt.LS and not opt.WGAN)

##################################

netG = NetG(ngf, nDep, total_z)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)

Gnets = [netG]
for net in [netD] + Gnets:
    try:
        net.apply(weights_init)
    except Exception as e:
        print(e, "weightinit")
    pass
    net = net.to(device)
    print(net)

if opt.useSavedWeights is not '':
    filename = opt.useSavedWeights
    netG.load_state_dict(torch.load(filename))
    netD.load_state_dict(torch.load(filename.replace('netG', 'netD')))
    text_file = open(opt.outputFolder + "options.txt", "a+")
    text_file.write('\nUsing Saved weight from: ' + str(opt.useSavedWeights))
    text_file.close()
    netG.train()
    netD.train()

if opt.addNoise:
    noise = torch.FloatTensor(opt.batchSize, noise_dim, noise_spatial_dim, noise_spatial_dim)
    fixnoise = torch.FloatTensor(opt.batchSize, noise_dim, noise_spatial_dim, noise_spatial_dim)
    noise = noise.to(device)
    fixnoise = fixnoise.to(device)
    noise = setNoise(noise)
    fixnoise = setNoise(fixnoise)

real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))  # netD.parameters()
optimizerG = optim.Adam([param for net in Gnets for param in list(net.parameters())], lr=opt.lr,
                        betas=(opt.beta1, 0.999))

cooc_l2_criterion = torch.nn.MSELoss()
cooc_kl_criterion = torch.nn.KLDivLoss(reduction='batchmean')
cooc_l1_criterion = torch.nn.L1Loss()
# -----------------------------------------------------------
start_time = time.time()
len_data = len(dataloader)
for epoch in range(1, opt.niter + 1):
    tqdm_obj = tqdm(enumerate(dataloader, 0), total=len_data, leave=False)
    using_weights = opt.useSavedWeights if opt.useSavedWeights is not '' else 'No'
    print('Epoch: {}, Image: {}, ImageSize: {}, Using Checkpoint: {}'.format(epoch, opt.texturePath, opt.imageSize,
                                                                             using_weights))
    if opt.useKLCooc:
        print("Using KL Divergence")
    for i, data in tqdm_obj:
        t0 = time.time()
        sys.stdout.flush()

        # Preprocess Data
        image_data, cooc_vol, cooc_gt, cooc_vol_D = data
        image_data = image_data.to(device)
        if not opt.coocNorm:
            cooc_vol = setCooc(cooc_vol)
            cooc_vol_D = setCooc(cooc_vol_D)
        cooc_vol = cooc_vol.to(device)
        cooc_vol_D = cooc_vol_D.to(device)
        cooc_vol_og = cooc_vol
        cooc_gt = cooc_gt.to(device)
        if opt.addNoise:
            noise = setNoise(torch.FloatTensor(opt.batchSize, noise_dim, noise_spatial_dim, noise_spatial_dim)).to(
                device)
            if opt.noiseFirst:
                cooc_vol = torch.cat((noise, cooc_vol), 1)
            else:
                cooc_vol = torch.cat((cooc_vol, noise), 1)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch

        netD.zero_grad()
        output = netD(image_data, cooc_vol_D)
        errD_real = criterion(output, output.detach() * 0 + real_label)
        errD_real.backward()
        D_x = output.mean()

        ## Train with all-fake batch
        # noise = setNoise(noise)  # Not needed for cooc
        fake = netG(cooc_vol)
        output = netD(fake.detach(), cooc_vol_D)
        errD_fake = criterion(output, output.detach() * 0 + fake_label)
        errD_fake.backward()

        D_G_z1 = output.mean()
        errD = errD_real + errD_fake
        if opt.WGAN:
            # TODO: Image size needs to be multiple of 32 for some reason :P
            gradient_penalty = calc_gradient_penalty(netD, cooc_vol_D, image_data, fake)  ##for case fewer text images
            gradient_penalty.backward()

        optimizerD.step()
        if i > 0 and opt.WGAN and i % opt.dIter != 0:
            continue  ##critic steps to 1 GEN steps

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################

        for net in Gnets:
            net.zero_grad()

        # noise = setNoise(noise)  # Not needed for cooc
        fake = netG(cooc_vol)
        output = netD(fake, cooc_vol_D)
        loss_adv = criterion(output, output.detach() * 0 + real_label)
        if opt.coocLoss:
            fake_min, fake_max = torch.min(fake), torch.max(fake)
            fake = 255.0 * (fake - fake_min) / (fake_max - fake_min)  # Have to normalize it for proper cooc calc
            loss_cooc = get_loss_for_cooc(fake, cooc_vol_og, cooc_gt)
            if math.isnan(loss_cooc.item()):
                print("Found a NAN in the calculation. Stopping Now! :(")
                exit()
            loss_value = loss_cooc.item()
        else:
            loss_cooc = 0
            loss_value = 0
        D_G_z2 = output.mean()
        errG = loss_adv + loss_cooc
        errG.backward()
        optimizerG.step()
        tqdm_obj.set_description("Epoch: {}, D(x): {:.3f}, D(G(z)): {:.3f}/{:.3f}, CLoss:{:.4f}".
                                 format(epoch, D_x, D_G_z1, D_G_z2, loss_value))

        ### RUN INFERENCE AND SAVE LARGE OUTPUT MOSAICS
        if epoch == 1 or epoch % 5 == 0 and i == 0:  # i == 0 is only to save the first output
            vutils.save_image(image_data, '%s/real_textures.jpg' % opt.outputFolder, normalize=True)
            vutils.save_image(fake, '%s/generated_textures_%03d_%s.jpg' % (opt.outputFolder, epoch, desc),
                              normalize=True)

            netG.eval()
            with torch.no_grad():
                if opt.addNoise:
                    if opt.noiseFirst:
                        fix_cooc = torch.cat((fixnoise, cooc_vol_og), 1)
                    else:
                        fix_cooc = torch.cat((cooc_vol_og, fixnoise), 1)
                else:
                    fix_cooc = cooc_vol_og
                fakeBig = netG(fix_cooc)

            vutils.save_image(fakeBig, '%s/big_texture_%03d_%s.jpg' % (opt.outputFolder, epoch, desc), normalize=True)
            netG.train()

            # OPTIONAL
            # save/load model for later use if desired
            outModelNameG = '%s/netG_%d_%s.pth' % (opt.outputFolder, epoch, desc)
            outModelNameD = '%s/netD_%d_%s.pth' % (opt.outputFolder, epoch, desc)
            torch.save(netG.state_dict(), outModelNameG)
            torch.save(netD.state_dict(), outModelNameD)

            plt.figure()
            plt.plot(G_losses)
            plt.title("Generator Losses")
            plt.savefig('%s/01_GeneratorLosses.jpg' % opt.outputFolder)
            plt.close()

            plt.figure()
            plt.plot(D_losses)
            plt.title("Discriminator Losses")
            plt.savefig('%s/01_DiscriminatorLosses.jpg' % opt.outputFolder)
            plt.close()

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

plt.figure()
plt.plot(G_losses)
plt.title("Generator Losses")
plt.savefig('%s/01_GeneratorLosses.jpg' % opt.outputFolder)
plt.close()

plt.figure()
plt.plot(D_losses)
plt.title("Discriminator Losses")
plt.savefig('%s/01_DiscriminatorLosses.jpg' % opt.outputFolder)
plt.close()

print("Total Time: ", time.time() - start_time)
print("Saved at: {}".format(opt.outputFolder))
