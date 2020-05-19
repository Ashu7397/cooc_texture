import os

import numpy as np
import scipy.ndimage as snd
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import Dataset

import cooc_utils
from config import opt


class TextureDataset(Dataset):
    """Dataset wrapping images from a random folder with textures

    Arguments:
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, img_path, factor_scale, transform=None, cooc_tr=None, epoch_constant=2000,
                 sampling_rate_kmeans=0.1):
        self.img_path = img_path
        self.image_transform = transform
        self.cooc_transform = cooc_tr
        self.factor = factor_scale
        self.f_cooc = cooc_utils.build_gaussian_filter(cooc_utils.WIN_SIZE, cooc_utils.SIGMA)
        self.k_clusters = opt.kVal
        self.sampling_rate = sampling_rate_kmeans
        self.im_tf = tf.ToTensor()
        if True:  ##ok this is for 1 worker only!
            names = [img_path] if os.path.isfile(img_path) else sorted(os.listdir(img_path))
            self.X_train = []
            self.Y_train = []
            for n in names:
                name = n if os.path.isfile(img_path) else self.img_path + n
                try:
                    img = Image.open(name)
                    try:
                        img = img.convert('RGB')  ##fixes truncation???
                    except:
                        pass
                except Exception as e:
                    print(e, name)
                    continue

                self.X_train += [img]
                print(n, "img added", img.size, "total length", len(self.X_train))
                if len(self.X_train) > 4000:
                    break  ##usually want to avoid so many files

        ##this affects epoch length..

        print('Evaluating GMM params...')
        _, im_gmm = cooc_utils.estimate_gmm(img, k=self.k_clusters, sample_rate=self.sampling_rate)
        self.im_gmm = im_gmm
        print('\t Done.')
        in_image_tensor = (self.im_tf(self.X_train[0]) * 255).unsqueeze(0)
        cooc_vol_inline = cooc_utils.cooc_finder_tensor(in_image_tensor, im_gmm=self.im_gmm, f_cooc=self.f_cooc,
                                                        k=self.k_clusters, requires_tensor=True)
        cooc_vol_inline = cooc_vol_inline.cpu().numpy()
        np.save(opt.outputFolder + 'cooc_vol', cooc_vol_inline)
        self.Y_train.append(cooc_vol_inline)
        n_height = np.array(img).shape[0] - opt.imageSize
        n_width = np.array(img).shape[1] - opt.imageSize

        if len(self.X_train) < epoch_constant:
            c = int(epoch_constant / len(self.X_train))
            self.X_train *= c
            self.Y_train *= c

        np.random.seed(opt.manualSeed)
        self.top_idx = np.random.random_integers(low=0, high=n_height, size=epoch_constant)
        np.random.seed(opt.manualSeed)
        self.left_idx = np.random.random_integers(low=0, high=n_width, size=epoch_constant)
        np.save(opt.outputFolder + 'top_idx', np.array(self.top_idx))
        np.save(opt.outputFolder + 'left_idx', np.array(self.left_idx))
        np.save(opt.outputFolder + 'color_data', np.expand_dims(np.array(im_gmm.mean), 0).astype(int))

    def __getitem__(self, index):
        img = np.array(self.X_train[index])
        cooc_vol = self.Y_train[index]
        image_crop = img[self.top_idx[index]:self.top_idx[index] + opt.imageSize,
                     self.left_idx[index]:self.left_idx[index] + opt.imageSize]
        cooc_crop_full = cooc_vol[self.top_idx[index]:self.top_idx[index] + opt.imageSize,
                         self.left_idx[index]:self.left_idx[index] + opt.imageSize]

        # Scaling down the cooc to match the dimensions of the paper
        cooc_crop = snd.zoom(cooc_crop_full, (self.factor, self.factor, 1))
        cooc_crop_D = snd.zoom(cooc_crop_full, (self.factor * 4, self.factor * 4, 1))

        if self.image_transform is not None:
            img2 = self.image_transform(image_crop)
        if self.cooc_transform is not None:
            cooc_vol2 = self.cooc_transform(cooc_crop)
            cooc_vol_D = self.cooc_transform(cooc_crop_D)

        image_tf = tf.ToTensor()
        data_im_tensor = image_tf(img) * 255
        cooc_gt, _ = cooc_utils.collect_cooc_batch(data_im_tensor.unsqueeze(0), gmm=self.im_gmm, f_cooc=self.f_cooc)
        return img2, cooc_vol2, cooc_gt, cooc_vol_D

    def __len__(self):
        return len(self.X_train)


##inplace set noise
def setNoise(noise):
    noise = noise.detach() * 1.0
    noise.normal_(-1, 1)  # uniform_(0, 1)
    return noise


def setCooc(cooc_vol):
    return 2 * cooc_vol - 1
