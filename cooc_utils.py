'''
Co-Occurence Utils
Ported from Itai Lang's code (Py2 + TensorFlow) to Py3 + Pytorch
'''

import time
from functools import reduce
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transform
from sklearn import cluster
from sklearn.decomposition import PCA
from tqdm import trange

from config import opt, nDep

EPSILON = 2e-10


class GmmModel:
    def __init__(self, n_clusters, mean=None, var=None, weight=None):
        self.k = n_clusters  # Number of Gaussians in the model

        if mean is not None:
            self.mean = mean
        else:
            self.mean = [None for _ in range(self.k)]

        if var is not None:
            self.var = var
        else:
            self.var = [None for _ in range(self.k)]

        if weight is not None:
            self.weight = weight
        else:
            self.weight = [None for _ in range(self.k)]


def build_gaussian_filter(wid, sigma=1):
    """
    Create a (symmetric) 2D Gaussian filter
    :param wid: Window size. The output kernel will be win_size*win_size
    :param sigma:  Gaussian std
    :return: g: The Gaussian filter
    """
    n = (wid - 1) / 2
    y, x = np.ogrid[-n:n + 1, -n:n + 1]
    g = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    sumh = g.sum()
    if sumh != 0:
        g /= sumh
    g = torch.FloatTensor(g).unsqueeze(0).unsqueeze(0).to(device)

    return g


def estimate_gmm(input_image, sample_image=None, k=32, sample_rate=0.1):
    """
    Estimate the distribution of an image as a Gaussian Mixture Model (GMM).
    The function uses k-means to find the Gaussians' canter and variance.
    The normalized cluster occurence is regarded as the Gaussians' weight.
    In addition, the image of cluster index and cluster center per pixel is computed.
    :param input_image: Input image for the model estimation.
    :param sample_image: Sample of the input image for the k-means stage
    :param k: number of clusters for the k-means stage
    :param sample_rate: percentage of input samples used for the k-means stage
    :return:

    Adapted to Python by Anna Darzi, Sept. 2018
    Based on Matlab based on code by Itai Lang
    """
    input_image = np.array(input_image).astype(np.float32)
    in_height, in_width = input_image.shape[0:2:]
    dim = 1 if input_image.ndim == 2 else 3
    n_pixels = in_height * in_width

    if sample_image is None:
        # reformat the image:
        data = np.reshape(input_image, [n_pixels, dim], order='F')

        # sample the image
        n_samples = np.round(sample_rate * n_pixels).astype(int)
        np.random.seed(100)  # for reproducibility
        # np.random.seed()
        sel = np.random.permutation(n_pixels)[0:n_samples]
        data = data[sel]
    else:
        data = sample_image

    # k-means:
    cc_kmeans = cluster.KMeans(n_clusters=k, tol=0).fit(data)
    cc = cc_kmeans.cluster_centers_
    cc_labels = cc_kmeans.labels_
    # cc = kmeans2(data, k, iter=10000)[0]

    # cluster input image to cluster centers:
    idx_image = -1 * np.ones([in_height, in_width], dtype=np.int)
    min_dist = np.inf * np.ones([in_height, in_width], dtype=np.int)
    for i in range(k):
        curr_dist = (input_image - cc[i]) ** 2
        if dim == 3:
            curr_dist = np.sum(curr_dist, axis=2)
        min_dist = np.minimum(curr_dist, min_dist)
        idx_image[curr_dist == min_dist] = i

    gmm_model = GmmModel(n_clusters=k)
    # calculate Gaussian weight - for each cluster
    cluster_count = np.bincount(idx_image.flatten())
    gmm_model.weight = cluster_count.astype(float) / n_pixels

    # calculate mean and variance and create cluster image
    # gmm_mean = np.zeros_like(cc)
    # gmm_var = np.zeros_like(cc)
    cc_image = -1 * np.ones_like(input_image)
    for i in range(k):
        cluster_indicator = (idx_image == i)  # .astype(np.uint8)
        cluster_samples = input_image[cluster_indicator]
        gmm_model.mean[i] = cluster_samples.mean(0)
        gmm_model.var[i] = np.var(cluster_samples, 0, ddof=1)

        cc_image[cluster_indicator] = gmm_model.mean[i]
        # cc_image[cluster_indicator] = i

    return cc_image, gmm_model


########################################################################################################################

def get_level_mask(input_image, level):
    """ OBSOLETE creates a mask for input image according to level
    :param input_image: input image to create a mask for
    :param level: level for the mask
    :return level_mask: equals 1 where 'input_image' equals 'level', and 0 otherwise
    """

    ones = torch.ones_like(input_image)
    zeros = torch.zeros_like(input_image)
    # level_mask = torch.where(torch.equal(input_image, ones * level), ones, zeros)
    level_mask = torch.where(input_image == level, ones, zeros)
    level_mask = torch.sum(level_mask, dim=1, keepdim=True).unsqueeze(2)

    return level_mask


def collect_cooc_batch(input_images, gmm, f_cooc, input_masks=None):
    """ collect co-occurrence statistics
    :param input_images: MAKE SURE VALUES ARE [0,255]
                        batch of input image to collect for
    :param gmm: gmm model of reference image
    :param f_cooc: spatial kernel for collecting co-occurrence
    :param input_masks: batch of binary masks for collecting co-occurrence

    :return cooc_mat: co-occurrence matrix with size (n_bins, n_bins)
    :return cooc_sum: normalization factor for co-occurrence matrix
    """
    input_images = input_images.to(device)
    batch_size = input_images.shape[0]

    cooc_mat = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)

    # number of clusters in the gmm model
    k = len(gmm.mean)

    # soft_assign = [get_level_mask(input_images, i) for i in range(k)]  # non differentiable collection
    soft_assign = [
        apply_exponential_kernel_batch(input_images, torch.from_numpy(gmm.mean[i]), torch.from_numpy(gmm.var[i])) for i
        in range(k)]

    soft_assign_tensor = torch.cat(soft_assign, dim=1)

    for i in range(k):
        level_i_mask = soft_assign[i]
        if input_masks is not None:
            # level_i_mask = level_i_mask * input_masks
            level_i_mask = torch.mul(level_i_mask, torch.unsqueeze(input_masks, dim=1))

        level_i_mask = torch.squeeze(level_i_mask, axis=1)

        # Check convolution dimensions - otherwise squeeze -> convolve -> expand dim 1
        cooc_weight = torch.nn.functional.conv2d(level_i_mask, weight=f_cooc,
                                                 padding=((WIN_SIZE - 1) // 2, (WIN_SIZE - 1) // 2))
        # cooc_weight = level_i_mask

        if input_masks is not None:
            # cooc_weight = cooc_weight * input_masks
            cooc_weight = torch.mul(cooc_weight, input_masks)

        cooc_weight = torch.unsqueeze(cooc_weight, axis=1)

        cooc_row = torch.sum(cooc_weight.repeat(1, k, 1, 1, 1) * soft_assign_tensor, dim=(2, 3, 4))
        cooc_mat = torch.cat((cooc_mat, cooc_row), dim=1)

    cooc_mat = cooc_mat[:, 1:cooc_mat.shape[1] + 1]

    cooc_mat = cooc_mat.view((batch_size, k, k))
    cooc_mat = cooc_mat + cooc_mat.permute(0, 2, 1)

    cooc_sum = torch.sum(cooc_mat, dim=(1, 2))
    cooc_mat = cooc_mat / torch.unsqueeze(torch.unsqueeze(cooc_sum, dim=1), dim=1)

    # clip values below 0:
    max_val = torch.max(cooc_mat).item()
    cooc_mat = torch.clamp(cooc_mat, min=0, max=max_val)
    return cooc_mat, cooc_sum


def apply_exponential_kernel_batch(input_images, mu, sigma_sq):
    """ apply an exponential kernel to an input image
    :param input_images: image to apply the kernel to
    :param mu: the center of the kernel
    :param sigma_sq: the (half) width of the kernel, squared

    :return output_image: the image after applying the kernel
    """
    batch_num = input_images.shape[0]
    input_images_expand = input_images.unsqueeze(1)
    if len(mu.shape) == 0:
        mu.unsqueeze(1)
    if len(sigma_sq.shape) == 0:
        sigma_sq.unsqueeze(1)

    mu_expand = (torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(
        mu.to(device), dim=0), dim=0), dim=-1), dim=-1)).repeat(batch_num, 1, 1, 1, 1)
    sigma_sq_expand = (torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(
        sigma_sq.to(device) + EPSILON, dim=0), dim=0), dim=-1), dim=-1)).repeat(batch_num, 1, 1, 1, 1)

    dist_sq = torch.sum(torch.pow((input_images_expand - mu_expand), 2) / sigma_sq_expand, axis=2, keepdims=True)
    output_image = torch.exp(-dist_sq)
    return output_image


def generate_images_crops(image, crop_size, stride=1):
    im_height, im_width = image.shape[0:2]
    crop_height, crop_width = crop_size[0:2]
    n_height = im_height - crop_height + 1
    n_width = im_width - crop_width + 1
    image_crops = []
    for i in range(0, n_height, stride):
        for j in range(0, n_width, stride):
            crop = image[i:i + crop_height, j:j + crop_width]
            image_crops.append(crop)
    return image_crops


def generate_image_tensor_crops(image, crop_size, stride=1):
    im_height, im_width = image.shape[2:4]
    crop_height, crop_width = crop_size[0:2]
    n_height = im_height - crop_height + 1
    n_width = im_width - crop_width + 1
    image_crops = []
    for i in range(0, n_height, stride):
        for j in range(0, n_width, stride):
            crop = image[:, :, i:i + crop_height, j:j + crop_width]
            image_crops.append(crop)
    return image_crops


def find_factors_of(n):
    # Finds the factors of a number
    return set(reduce(list.__add__,
                      ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))


def find_best_batch_size(number_of_crops, max_batch_size=4000):
    # Find best batch size such that we do not lose any crops
    factors_crops = find_factors_of(number_of_crops)
    nextLowest = lambda seq, x: min([(x - i, i) for i in seq if x >= i] or [(0, None)])
    batch_cooc = nextLowest(factors_crops, max_batch_size)[1]
    return batch_cooc


def process_crops_for_cooc(crops, batch_size, gmm_in, f_cooc, k=8, require_tensor=False, input_numpy=False):
    n_crops = len(crops)
    batch_start = 0
    batch_end = batch_start + batch_size
    n_batches = n_crops // batch_size
    cooc_vecs = []
    print('\tProcessing batches: {}'.format(n_batches))
    for counter in trange(n_batches):
        crops_batch = crops[batch_start:batch_end]
        if input_numpy:
            crops_batch = [(image_tf(image_crop) * 255).to(device) for image_crop in crops_batch]
            crops_batch = torch.stack(crops_batch, axis=0).squeeze(0)  # TODO: Check if axis is 1 or 0
        else:
            crops_batch = torch.stack(crops_batch, axis=1).squeeze(0)

        cooc_batch, _ = collect_cooc_batch(crops_batch, gmm=gmm_in, f_cooc=f_cooc)
        if require_tensor:
            cooc_batch_vecs = [cooc_batch[n].view(k ** 2) for n in range(batch_size)]
        else:
            cooc_batch_vecs = [np.reshape(cooc_batch[n].cpu().numpy(), k ** 2) for n in range(batch_size)]

        cooc_vecs += cooc_batch_vecs

        batch_start = batch_end
        batch_end += batch_size
    return cooc_vecs


def cooc_finder(original_image, image_folder=None, in_image_name=None, do_save=False, im_gmm=None, f_cooc=None,
                requires_tensor=False, k=8, max_batch=4000):
    original_image_shape = original_image.shape
    im = np.pad(original_image,
                ((PATCH_SIZE // 2 - 1, PATCH_SIZE // 2), (PATCH_SIZE // 2, PATCH_SIZE // 2 - 1), (0, 0)),
                mode='symmetric')
    if f_cooc is None:
        f_cooc = build_gaussian_filter(WIN_SIZE, SIGMA)
    # f_cooc = np.expand_dims(np.expand_dims(f_cooc, axis=0), axis=0)
    crops = generate_images_crops(im, (PATCH_SIZE, PATCH_SIZE), STRIDE_VAL)
    print('Number of patches: ', len(crops))

    batch_cooc = find_best_batch_size(len(crops), max_batch)
    print('The optimal batch size is :', batch_cooc)

    if im_gmm is None:
        print('Evaluating GMM params...')
        _, im_gmm = estimate_gmm(im, k=k, sample_rate=SAMPLE_RATE)
        print('\t Done.')

    print('Initializing co-occurrence calculations model.')

    processed_crops = process_crops_for_cooc(crops, batch_cooc, gmm_in=im_gmm, f_cooc=f_cooc, k=k,
                                             require_tensor=requires_tensor, input_numpy=True)
    if requires_tensor:
        reshaped_cooc = torch.cat(processed_crops, 0).view(
            (original_image_shape[0] // STRIDE_VAL, original_image_shape[1] // STRIDE_VAL, k * k))
    else:
        reshaped_cooc = np.array(processed_crops).reshape(
            (original_image_shape[0] // STRIDE_VAL, original_image_shape[1] // STRIDE_VAL, k * k))

    if do_save:
        saved_cooc_name = 'cooc_data/' + image_folder + '_' + in_image_name + '_'
        np.save(saved_cooc_name + 'reshaped', reshaped_cooc)

    return reshaped_cooc


def downsample_to_proportion(rows, proportion=1):
    return list(islice(rows, 0, len(rows), int(1 / proportion)))


def cooc_finder_tensor(in_image, im_gmm=None, f_cooc=None, k=8, requires_tensor=True, max_batch=4000, in_network=False):
    """
    Args:
        in_image: Tensor image of shape [num_of_imag, channels, xdim, ydim]
        im_gmm: Object of gmm model
        f_cooc: Gaussian filter
        k: Number of cluster of computation
        requires_tensor: Do we need the output to be a tensor too

    Returns:

    """
    original_image_shape = in_image.shape
    num_of_images = original_image_shape[0]
    batched_imgs = []
    size_sampled = int(opt.imageSize / 2 ** nDep)
    size_sampled_sq = int(size_sampled ** 2)
    for curr_image in range(num_of_images):
        original_image = in_image[curr_image, :, :, :].unsqueeze(0)
        im = torch.nn.functional.pad(original_image,
                                     (PATCH_SIZE // 2 - 1, PATCH_SIZE // 2, PATCH_SIZE // 2, PATCH_SIZE // 2 - 1),
                                     mode='reflect')
        crops = generate_image_tensor_crops(im, (PATCH_SIZE, PATCH_SIZE), STRIDE_VAL)
        if in_network:
            crops = downsample_to_proportion(crops, size_sampled_sq / len(crops))
        print('Number of patches: ', len(crops))

        batch_cooc = find_best_batch_size(len(crops), max_batch)
        print('The optimal batch size is :', batch_cooc)
        if f_cooc is None:
            f_cooc = build_gaussian_filter(WIN_SIZE, SIGMA)
        if im_gmm is None:
            print('Evaluating GMM params...')
            _, im_gmm = estimate_gmm(im.cpu(), k=k, sample_rate=SAMPLE_RATE)
            print('\t Done.')
        print('Initializing co-occurrence calculations model.')
        # crops = torch.cat(crops, 0)
        processed_crops = process_crops_for_cooc(crops, batch_cooc, gmm_in=im_gmm, f_cooc=f_cooc, k=k,
                                                 require_tensor=requires_tensor)
        if in_network:
            reshaped_cooc = torch.cat(processed_crops, 0).view(
                (k * k, size_sampled // STRIDE_VAL, size_sampled // STRIDE_VAL))
        else:
            reshaped_cooc = torch.cat(processed_crops, 0).view(
                (original_image_shape[2] // STRIDE_VAL, original_image_shape[3] // STRIDE_VAL, k * k))
        batched_imgs.append(reshaped_cooc.unsqueeze(0))
    if in_network:
        return torch.cat(batched_imgs)
    return reshaped_cooc


def do_PCA_main(data_im, file_name, cooc_vol=None):
    pca_dimensions = 1
    pca = PCA(n_components=pca_dimensions)

    cooc_time = time.time()
    if cooc_vol is None:
        cooc_volume = cooc_image_finder(data_im)
    else:
        cooc_volume = cooc_vol
    print("Found Cooc Volume in ", time.time() - cooc_time)
    x_shape, y_shape, channels = cooc_volume.shape
    pca_image = np.zeros((x_shape, y_shape, pca_dimensions))
    pca_var_values = np.zeros(pca_image.shape[0])
    for i, matrix in enumerate(cooc_volume):
        value = pca.fit_transform(matrix)
        pca_image[i, :] = value
        pca_var_values = sum(pca.explained_variance_ratio_)
    pca_image_norm = (pca_image - np.min(pca_image)) / (np.max(pca_image) - np.min(pca_image))
    plt.figure()
    plt.subplot(211)
    plt.title('Original Image: ' + file_name)
    plt.imshow(data_im.astype(int))
    plt.draw()
    plt.subplot(212)
    plt.title('CooC PCA Normalized with min/max variance: ' + "{:.2f}".format(np.min(pca_var_values)) + ' / '
              + "{:.2f}".format(np.max(pca_var_values)))
    print(pca_image_norm.shape)
    plt.imshow(pca_image_norm[:, :, 0], cmap='plasma')
    figure_plotted = plt.gcf()
    plt.show()
    # figure_plotted.savefig('./pcaResults/' + file_name, bbox_inches='tight')


SAMPLE_RATE = 0.1  # for gmm estimation
WIN_SIZE = 51
SIGMA = np.sqrt(WIN_SIZE)
BATCH_SIZE_CROPS = 64
PATCH_SIZE = 64
K_VAL = 2
STRIDE_VAL = 1
TEST = 1
image_tf = transform.ToTensor()

torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
if __name__ == '__main__':
    start_time = time.time()
    image_type_name = 'prime'
    folder_name = '/mnt/storage/PyCharmUnoff/FCN_CGAN/samples/' + image_type_name + '/'
    image_name = [im_name for im_name in os.listdir(folder_name) if '.png' in im_name or '.jpg' in im_name][0]
    data_im = mpimg.imread(folder_name + image_name, format=np.float32)
    image_name_in = 'K=' + str(K_VAL) + '_' + image_name.split('.')[0]

    if TEST:
        py_cooc = np.load('cooc_data/DTD_Pytorch_K=2_honeycombed_0074_reshaped.npy')
        tf_cooc = np.load('cooc_data/DTD_TF_K=2_honeycombed_0074_reshaped.npy')
        new_py = py_cooc.sum(axis=2)
        new_tf = tf_cooc.sum(axis=2)
        do_PCA_main(data_im, 'Pytorch', py_cooc)
        do_PCA_main(data_im, 'TF', tf_cooc)
        if (np.array_equal(py_cooc, tf_cooc)):
            print(np.sum(np.abs(py_cooc-tf_cooc)))
            print("Ho gaya kaam")
        else:
            print('Difference: ', np.sum(np.abs(py_cooc-tf_cooc)))
            print("Not very accurate")
    else:
        cooc_vol = cooc_finder(data_im, 'DTD_Pytorch', image_name_in, do_save=True, k=K_VAL)

        # print('Evaluating GMM params...')
        # _, im_gmm = estimate_gmm(data_im, k=K_VAL, sample_rate=SAMPLE_RATE)
        # print('\t Done.')
        # print(im_gmm.mean)
        # f_c = build_gaussian_filter(WIN_SIZE, SIGMA)
        # 
        # data_im_tensor = image_tf(data_im)*255
        # cooc_vol, _= collect_cooc_batch(data_im_tensor.unsqueeze(0).to(device), gmm=im_gmm, f_cooc=f_c)
        # cooc_im = cooc_vol.squeeze(0).cpu().numpy()
        # plt.imshow(cooc_im, cmap='plasma')
        # plt.show()
        print('Done!!\nTotal Time Taken: ', time.time() - start_time)
'''
