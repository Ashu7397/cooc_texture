import os
from random import randint, choice

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as snd
import torch
import torchvision.utils as vutils
from PIL import Image

import cooc_utils
from config import opt, nDep, DEVICE, TRANSFORM_TEX, TRANSFORM_TEX_COOC
from network import NetG
from utils import setNoise, setCooc


def get_model_input(noise_vec, cooc_vec):
    '''
    
    Args:
        noise_vec: The noise vector to be concatenated (PyTorch Tensor)
        cooc_vec: The Co-Occurrence volume to be concatenated (PyTorch Tensor)

    Returns:

    '''
    if opt.addNoise:
        if opt.noiseFirst:
            model_input = torch.cat((noise_vec, cooc_vec), 1).to(DEVICE)
        else:
            model_input = torch.cat((cooc_vec, noise_vec), 1).to(DEVICE)
    else:
        model_input = cooc_vec
    return model_input


def get_box_around_image(image_crop, top_idx=None, left_idx=None, crop_size=None, color=[0, 255, 0], line_width=2,
                         is_crop=True):
    '''
    Takes in an image or image crop and draws a box of linewidth around it.
    Args:
        image_crop: The image/image_crop we want to draw a box around
        top_idx: The top index of the crop
        left_idx: The left index of the crop (Hence helps us to get get top-left corner)
        crop_size: The size of the box of crop with the given top-left corner
        color: The color of the box drawn around the image
        line_width: Thw width of the color border
        is_crop: Is the input image a crop, or the main sample image

    Returns: The input image with the boxes drawn at the corresponding locations

    '''
    line_color = np.reshape(color, (1, 1, 3))

    # Set all four sides' coordinates
    if is_crop:
        top = 0
        left = 0
    else:
        top = top_idx
        left = left_idx
    bottom = top + crop_size
    right = left + crop_size

    # Draw lines:
    if is_crop:
        image_crop[:, left:left + line_width, :] = line_color
        image_crop[:, right - line_width:right, :] = line_color
        image_crop[top:top + line_width, :, :] = line_color
        image_crop[bottom - line_width:bottom, :, :] = line_color
    else:
        image_crop[top:bottom, left:left + line_width, :] = line_color
        image_crop[top:bottom, right - line_width:right, :] = line_color
        image_crop[top:top + line_width, left:right, :] = line_color
        image_crop[bottom - line_width:bottom, left:right, :] = line_color

    return image_crop


def get_box_around_image_tensor(image_crop, crop_size, color=[0.0, 1.0, 0.0], line_width=6):
    '''
    Another function to draw a box around crop, but instead of numpy array, its a PyTorch Tensor
    Args:
        image_crop: The crop of image we want the box around
        crop_size: The size of the crop of input image
        color: The color of the box that we want
        line_width: The widht of the line border around the box

    Returns: Returns the image crop with the a box around it with color and linewidth given as input

    '''
    line_color = torch.from_numpy(np.array(color)).view((3, 1, 1))

    # Set all four sides' coordinates
    top = 0
    left = 0
    bottom = top + crop_size
    right = left + crop_size

    # Draw lines:
    image_crop[:, :, left:left + line_width] = line_color
    image_crop[:, :, right - line_width:right] = line_color
    image_crop[:, top:top + line_width, :] = line_color
    image_crop[:, bottom - line_width:bottom, :] = line_color

    return image_crop


def get_random_crop(data_image, cooc_vol, crop_size, function_name='Unknown', get_idx=False):
    '''
    Takes in entire image and Co-Occurence volume and gets out a crop of size crop_size
    Args:
        data_image: The entire input image from which we want to take crops from
        cooc_vol: The entire Co-Occurence Volume fom where we want to take the crops
        crop_size: The size of the crops we want during evaluation
        function_name: The name of the function which is calling, for easier debugging
        get_idx: Whether we want the top and left index to be returned

    Returns: The image crop along with cooc crop of crop size with index location if needed

    '''
    image_shape = data_image.shape
    top_idx, left_idx = randint(0, image_shape[0] - crop_size), randint(0, image_shape[1] - crop_size)
    print("Getting Random crop for {} with Top Idx: {}, Left Idx: {}".format(function_name, top_idx, left_idx))

    image_crop = data_image[top_idx:top_idx + crop_size, left_idx:left_idx + crop_size]
    cooc_crop = cooc_vol[top_idx:top_idx + crop_size, left_idx:left_idx + crop_size]
    if get_idx:
        return image_crop, cooc_crop, top_idx, left_idx
    return image_crop, cooc_crop


def process_crop_for_tensor(image_crop=None, cooc_crop=None, factor_scale=1):
    '''
    Takes in numpy arrays and converts them into PyTorch tensors which are normalized and scaled as required
    Args:
        image_crop: The image crop which we want to convert to tensor
        cooc_crop: The Co-Occurrence volume crop which we want to convert to tensor
        factor_scale: The scaling down factor of the co-occurence volume to match the crop_size at generation

    Returns: PyTorch tensors of the image crop and Co-Occurence volume normalized according to the requirements

    '''
    image_crop_tensor, cooc_crop_tensor = None, None
    if image_crop is not None:
        image_crop_tensor = TRANSFORM_TEX(image_crop).unsqueeze(0).to(DEVICE)
    if cooc_crop is not None:
        if factor_scale != 1:
            cooc_crop = snd.zoom(cooc_crop, (factor_scale, factor_scale, 1))
        if not opt.coocNorm:
            cooc_crop = setCooc(cooc_crop)
        cooc_crop_tensor = TRANSFORM_TEX_COOC(cooc_crop).unsqueeze(0).to(DEVICE)

    return image_crop_tensor, cooc_crop_tensor


def interpolate_points_with_extrapolation(v_start, v_end, n_interp=8, n_extrap=0):
    '''
    Takes in the beginning vector and end vector to find interpolation and extrapolation between them
    Args:
        v_start: The start vector of interpolation
        v_end: The end point of the interpolation
        n_interp: The number of interpolation steps
        n_extrap: The number of extrapolation steps

    Returns: List of vectors interpolated and extrapolated with given number of steps

    '''
    n_in_ex = n_interp + 2 * n_extrap
    alpha = np.linspace(0, 1, num=n_interp, dtype=np.float32)
    step = alpha[1] - alpha[0]
    alpha_ext_pre = np.linspace(start=-step * n_extrap, stop=-step, num=n_extrap, dtype=np.float32)
    alpha_ext_post = np.linspace(start=1 + step, stop=1 + step * n_extrap, num=n_extrap, dtype=np.float32)
    alpha = np.concatenate((alpha_ext_pre, alpha, alpha_ext_post))

    v_size = v_start.size

    v_interpolated = []
    for j in range(n_in_ex):
        # Interpolate v_start and v_end
        a = alpha[j]
        temp = (1 - a) * v_start + a * v_end
        v_interpolated.append(temp)

    return v_interpolated


def naive_rgb_interpolation(data_image, top_idx, left_idx, crop_size, image_name, dir_to_save,
                            number_of_samples=8, extra=3):
    '''
    Takes in an images, takes crops out it and generates interpolation and extrapolation samples in the rgb space,
    and save them as a single image in the dir_to_save
    Args:
        data_image: The input main image to take crops from
        top_idx: The top index location of the crop we want
        left_idx: The left index location of the crop we want
        crop_size: The size of the crop we want for evaluation
        image_name: The name of the image we are running analysis on to save it
        dir_to_save: The directory where we want to save the image
        number_of_samples: The number of interpolation steps
        extra: The number of extrapolation steps

    Returns: Saves an image with interpolation and extrapolation between the two crops in the RGB space.

    '''
    image_crop1 = data_image[top_idx[0]:top_idx[0] + crop_size, left_idx[0]:left_idx[0] + crop_size]
    image_crop2 = data_image[top_idx[1]:top_idx[1] + crop_size, left_idx[1]:left_idx[1] + crop_size]
    images_interp_list = interpolate_points_with_extrapolation(image_crop1, image_crop2, number_of_samples, extra)
    total_len = len(images_interp_list)
    new_images_interp_list = []
    for counter in range(total_len):
        new_images_interp_list.append(get_box_around_image(images_interp_list[counter], crop_size=crop_size,
                                                           line_width=2, is_crop=True, color=[0, 0, 0]))
    new_images_interp_list[extra] = get_box_around_image(new_images_interp_list[extra], crop_size=crop_size,
                                                         line_width=4, is_crop=True)
    new_images_interp_list[total_len - extra - 1] = get_box_around_image(new_images_interp_list[total_len - extra - 1],
                                                                         crop_size=crop_size, line_width=4,
                                                                         is_crop=True)

    imgs_comb = np.hstack(list((np.asarray(i) for i in new_images_interp_list)))
    imgs_comb = imgs_comb.clip(0, 255)
    input_to_norm_min, input_to_norm_max = np.min(imgs_comb), np.max(imgs_comb)
    imgs_comb_norm = (imgs_comb - input_to_norm_min) / (input_to_norm_max - input_to_norm_min)
    plt.imsave(
        '%s/Naive_RGB_Interp_%03d_%s_%s_%d_%s_%d.jpg' % (dir_to_save, crop_size, image_name, 'RS:', opt.manualSeed,
                                                         'NS:', number_of_samples), imgs_comb_norm)


def noise_diversity(data_image, cooc_vol, latent_noise_dim, spatial_noise_dim, crop_size, image_name, dir_to_save,
                    trained_gan_network, number_of_samples=8, color=[255, 0, 0], top_idx=None, left_idx=None):
    '''
    Takes in the sample image, extracts a random crop(or one with given coordinates) of the co-occurrence volume and
    generates images with concatenating different random noise vectors
    Args:
        data_image: The main sample image to run analysis on
        cooc_vol: The entire Co-Occurrence volume of the sample image (w*h*k**2)
        latent_noise_dim: THe latent dimension of the noise vector
        spatial_noise_dim: The spatial dimension of the noise vector to be concatenated
        crop_size: The image size of the analysis
        image_name: Name of the image we are running evaluation on
        dir_to_save: The directory to save the output of this evaluation
        trained_gan_network: The trained generator network for generation of new images
        number_of_samples: Number of interpolation steps
        color: Color of the box we want around the input crop
        top_idx: The top index location of the crop in pixel
        left_idx: The left index location of the crop in pixel

    Returns: Saves an image with the diversity in the generated images due to change in the noise vector
    for the same Co-Occurrence vector

    '''
    if top_idx is None:
        image_crop, cooc_crop, top_idx, left_idx = get_random_crop(data_image, cooc_vol, crop_size, 'Noise Diversity',
                                                                     True)
    else:
        image_crop = data_image[top_idx:top_idx + crop_size, left_idx:left_idx + crop_size]
        cooc_crop = cooc_vol[top_idx:top_idx + crop_size, left_idx:left_idx + crop_size]

    image_crop = get_box_around_image(image_crop, top_idx, left_idx, crop_size, color)
    image_crop_tensor, cooc_crop_tensor = process_crop_for_tensor(image_crop, cooc_crop, spatial_noise_dim / crop_size)

    cooc_vol = torch.cat(number_of_samples * [cooc_crop_tensor])
    if opt.addNoise:
        noise_tensor = setNoise(
            torch.FloatTensor(number_of_samples, latent_noise_dim, spatial_noise_dim, spatial_noise_dim)).to(DEVICE)

    model_input = get_model_input(noise_tensor, cooc_vol)

    with torch.no_grad():
        diversity_output = trained_gan_network(model_input)
    diversity_output = torch.cat((image_crop_tensor, diversity_output), 0)
    vutils.save_image(diversity_output, '%s/Diversity_Noise_%03d_%s_%s_%d_%d_%d.jpg' %
                      (dir_to_save, crop_size, image_name, 'RS:', opt.manualSeed, top_idx, left_idx),
                      normalize=True, nrow=number_of_samples + 1)
    return top_idx, left_idx


def get_cooc_interps(data_image, cooc_vol, scale_factor, crop_size, number_of_samples, get_image=False, extrapolation=0,
                     top_idx=None, left_idx=None):
    '''
    Takes in the input image with the entire Co-Occurence volume to take crops from and find interpolation and
    extrapolation between the Co-Occurrence crops and return them as a list
    Args:
        data_image: The main sample image to run analysis on
        cooc_vol: The entire Co-Occurrence volume of the sample image (w*h*k**2)
        scale_factor: The factor to sclae down the Co-Occurrence volume to match the imageSize we want at output
        crop_size: The image size of the evaluation
        number_of_samples: Number of interpolation steps
        get_image: Whether we want the interpolated images as output in list
        extrapolation: The steps of extrapolation on either side
        top_idx: The top index location of the crop in pixel
        left_idx: The left index location of the crop in pixel

    Returns: List of interpolated and extrapolated Co-Occurrence crops and image crops if needed

    '''
    cooc_crops = []
    image_crops = []
    for counter in range(2):
        if top_idx is None:
            image_crop, cooc_crop = get_random_crop(data_image, cooc_vol, crop_size, 'Get_Cooc_Interps')
        else:
            image_crop = data_image[top_idx[counter]:top_idx[counter] + crop_size,
                         left_idx[counter]:left_idx[counter] + crop_size]
            cooc_crop = cooc_vol[top_idx[counter]:top_idx[counter] + crop_size,
                        left_idx[counter]:left_idx[counter] + crop_size]
        image_crop_tensor, cooc_crop_tensor = process_crop_for_tensor(image_crop, cooc_crop, scale_factor)

        image_crops.append(image_crop_tensor)
        cooc_crops.append(cooc_crop_tensor)

    interp_cooc = interpolate_points_with_extrapolation(cooc_crops[0], cooc_crops[1], number_of_samples, extrapolation)
    if get_image:
        image_crops = interpolate_points_with_extrapolation(image_crops[0], image_crops[1], number_of_samples,
                                                            extrapolation)
    return interp_cooc, image_crops


def written_texture(data_image, cooc_vol, scale_factor,
                   crop_size, image_name, dir_to_save, trained_gan_network):
    '''

    Args:
        data_image: The main sample image to run analysis on
        cooc_vol: The entire Co-Occurrence volume of the sample image (w*h*k**2)
        scale_factor: The factor to sclae down the Co-Occurrence volume to match the imageSize we want at output
        crop_size: The image size of the evaluation
        image_name: The name of the input image
        dir_to_save: The directory where we want to save the image
        trained_gan_network: The trained generator model

    Returns: Saves an image with the something written on top of it

    '''
    # RS: 53626
    size = [11, 22]  # Size of block in coord dimention
    top_idx, left_idx = [200, 0], [150, 100]  # Coord of cooc. First one is the base texture, and second is the diff_cooc

    image_crop1 = data_image[top_idx[0]:top_idx[0] + crop_size, left_idx[0]:left_idx[0] + crop_size]
    cooc_crop1 = cooc_vol[top_idx[0]:top_idx[0] + crop_size, left_idx[0]:left_idx[0] + crop_size]

    image_crop2 = data_image[top_idx[1]:top_idx[1] + crop_size, left_idx[1]:left_idx[1] + crop_size]
    cooc_crop2 = cooc_vol[top_idx[1]:top_idx[1] + crop_size, left_idx[1]:left_idx[1] + crop_size]

    image_crop1, cooc_crop1 = process_crop_for_tensor(image_crop1, cooc_crop1, scale_factor)
    image_crop2, cooc_crop2 = process_crop_for_tensor(image_crop2, cooc_crop2, scale_factor)

    entire_block = torch.cat([cooc_crop1] * size[1], dim=3)
    entire_block = torch.cat([entire_block] * size[0], dim=2)

    entire_block = get_2020_block(entire_block, cooc_crop2)
    noise_in = setNoise(torch.FloatTensor(entire_block.shape).to(DEVICE))
    model_input = get_model_input(noise_in, entire_block)
    with torch.no_grad():
        block_out = trained_gan_network(model_input)
    vutils.save_image(block_out, '%s/%s_Teaser_%03d_%s_%s%d.jpg' %
                      (dir_to_save, '2020', crop_size, image_name, 'RS:', opt.manualSeed), normalize=True, pad_value=1)


def get_2020_block(block, diff_cooc):
    '''
    Takes in a block of canvas and 'paints'(replaces the co-oc with the diff_cooc) the numbers 2020 on it.
    Args:
        block: Takes entire canvas for 'painting' the texture image
        diff_cooc: The different Co-Occurrence we want to 'paint' with

    Returns: The block with the text printed on it

    '''
    cs = diff_cooc.shape[2]

    #### PART 1 #####
    block[:, :, 3 * cs:4 * cs, 3 * cs:4 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)
    block[:, :, 3 * cs:4 * cs, 8 * cs:9 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)
    block[:, :, 3 * cs:4 * cs, 13 * cs:14 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)
    block[:, :, 3 * cs:4 * cs, 18 * cs:19 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)

    #### PART 2 ####
    block[:, :, 4 * cs:5 * cs, 2 * cs:3 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)
    block[:, :, 4 * cs:5 * cs, 4 * cs:5 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)
    block[:, :, 4 * cs:5 * cs, 7 * cs:8 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)
    block[:, :, 4 * cs:5 * cs, 9 * cs:10 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)
    block[:, :, 4 * cs:5 * cs, 12 * cs:13 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)
    block[:, :, 4 * cs:5 * cs, 14 * cs:15 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)
    block[:, :, 4 * cs:5 * cs, 17 * cs:18 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)
    block[:, :, 4 * cs:5 * cs, 19 * cs:20 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)

    #### PART 3 ####
    block[:, :, 5 * cs:6 * cs, 4 * cs:5 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)
    block[:, :, 5 * cs:6 * cs, 7 * cs:8 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)
    block[:, :, 5 * cs:6 * cs, 9 * cs:10 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)
    block[:, :, 5 * cs:6 * cs, 14 * cs:15 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)
    block[:, :, 5 * cs:6 * cs, 17 * cs:18 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)
    block[:, :, 5 * cs:6 * cs, 19 * cs:20 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)

    #### PART 4 ####
    block[:, :, 6 * cs:7 * cs, 3 * cs:4 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)
    block[:, :, 6 * cs:7 * cs, 7 * cs:8 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)
    block[:, :, 6 * cs:7 * cs, 9 * cs:10 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)
    block[:, :, 6 * cs:7 * cs, 13 * cs:14 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)
    block[:, :, 6 * cs:7 * cs, 17 * cs:18 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)
    block[:, :, 6 * cs:7 * cs, 19 * cs:20 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)

    #### PART 5 #####
    block[:, :, 7 * cs:8 * cs, 2 * cs:5 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 3, dim=3)
    block[:, :, 7 * cs:8 * cs, 8 * cs:9 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)
    block[:, :, 7 * cs:8 * cs, 12 * cs:15 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 3, dim=3)
    block[:, :, 7 * cs:8 * cs, 18 * cs:19 * cs] = torch.cat([torch.cat([diff_cooc], dim=2)] * 1, dim=3)

    return block


def cooc_interpolation_and_extrap(data_image, cooc_vol, scale_factor, noise_in, crop_size, image_name, dir_to_save,
                                  trained_gan_network, number_of_samples=8, extra=3, top_idx=None, left_idx=None):
    '''
    Takes an input image, takes 2 random crops, finds the naive rgb interpolation and extrapolations in the image space.
    Does the same interpolation and extrapolation in the Co-Occurrence domain, and passes this through the generator to
    provide a stable and smooth output
    Args:
        data_image: The main sample image to run analysis on
        cooc_vol: The entire Co-Occurrence volume of the sample image (w*h*k**2)
        scale_factor: The scale down factor for cooc volume
        noise_in: Noise vector to be concatenated to cooc
        crop_size: The image size of the analysis
        image_name: Name of the image we are running evaluation on
        dir_to_save: The directory to save the output of this evaluation
        trained_gan_network: The trained generator network for generation of new images
        number_of_samples: Number of interpolation steps
        extra: The number of extrapolation steps on wither side
        top_idx: Top index of crop (Array of 2 here)
        left_idx: Left index of crop (Array of 2 here)

    Returns: Saves an image with _number_of_samples_ interpolation steps and _extra_ steps of
    extrapolation in the cooc_domain.

    '''
    if top_idx is None:
        top_idx, left_idx = [], []
        _, _, top_idx1, left_idx1 = get_random_crop(data_image, cooc_vol, crop_size, 'Cooc Interp_Extrap', get_idx=True)
        _, _, top_idx2, left_idx2 = get_random_crop(data_image, cooc_vol, crop_size, 'Cooc Interp_Extrap', get_idx=True)
        top_idx.extend((top_idx1, top_idx2))
        left_idx.extend((left_idx1, left_idx2))
    naive_rgb_interpolation(data_image, top_idx, left_idx, crop_size, image_name, dir_to_save,
                            number_of_samples, extra)
    cooc_shape = cooc_vol.shape[0:2]
    if crop_size < min(cooc_shape):
        cooc_interp_tensor, image_crops_tensor = get_cooc_interps(data_image, cooc_vol, scale_factor, crop_size,
                                                                  number_of_samples, extrapolation=extra,
                                                                  top_idx=top_idx, left_idx=left_idx)
        cooc_interp_tensor = torch.cat(cooc_interp_tensor)
    else:
        print("Crop size bigger than image size in interp")
    if opt.addNoise:
        noise_tensor = setNoise(torch.cat((number_of_samples + 2 * extra) * [noise_in]))
    model_input = get_model_input(noise_tensor, cooc_interp_tensor)
    with torch.no_grad():
        cooc_interp_out = trained_gan_network(model_input)
    total_len = len(cooc_interp_out)

    #### Printing and Saving #######
    cooc_interp_out[extra] = get_box_around_image_tensor(cooc_interp_out[extra], crop_size=crop_size)
    cooc_interp_out[total_len - extra - 1] = get_box_around_image_tensor(cooc_interp_out[total_len - extra - 1],
                                                                         crop_size=crop_size)

    vutils.save_image(cooc_interp_out, '%s/Interp_Cooc_%03d_%s_%s_%d_%s_%d.jpg' %
                      (dir_to_save, crop_size, image_name, 'RS:', opt.manualSeed, 'NS:', number_of_samples),
                      normalize=True, nrow=(number_of_samples + 2 * extra) + 2, pad_value=1, padding=2)


def fidelity_diversity(data_image, cooc_vol, latent_noise_dim, spatial_noise_dim, crop_size, image_name, dir_to_save,
                       trained_gan_network, number_of_samples=8):
    # Wrapper for diversity with diff crops
    top_idx_arr, left_idx_arr = [], []
    for counter in range(number_of_samples // 2):
        temp_image = np.copy(data_image)
        top_idx, left_idx = noise_diversity(temp_image, cooc_vol, latent_noise_dim, spatial_noise_dim, crop_size,
                                            image_name,
                                            dir_to_save, trained_gan_network, number_of_samples=8,
                                            color=COLOR_ARRAY[counter])
        top_idx_arr.append(top_idx)
        left_idx_arr.append(left_idx)
    for counter in range(number_of_samples // 2):
        data_image = get_box_around_image(data_image, top_idx_arr[counter], left_idx_arr[counter], crop_size,
                                          COLOR_ARRAY[counter], line_width=4, is_crop=False)
    plt.imsave(dir_to_save + 'Fidelity+Diversity.png', data_image)


def get_folder_details():
    """

    Returns: The trained generator network read from modelPath directory, the input image as a numpy array,
    the Co-Occurrence volume calculated on the data image, the name of image and the gmm model along with the
    gaussian filter used to calculate that

    """
    gen_network = NetG(opt.ngf, nDep, (opt.kVal * opt.kVal) * 2)
    sample_image_filename = opt.texturePath
    results_folder = opt.modelPath
    epoch = opt.checkpointNumber
    try:
        location_string = results_folder + 'netG_' + str(epoch) + '_fc1.0_ngf120_ndf120_dep5-5_WGAN.pth'
        gen_network.load_state_dict(torch.load(location_string))
        print(location_string)
    except FileNotFoundError:
        print("Please check your modelPath along with the checkpointNumber arguments!")
    network_trained = gen_network.to(DEVICE).eval()
    try:
        data_image = np.array(Image.open(sample_image_filename).convert('RGB'))
    except:
        print('Check your texturePath argument and make sure its the sample image!')
    f_gauss = cooc_utils.build_gaussian_filter(cooc_utils.WIN_SIZE, cooc_utils.SIGMA)
    print('Evaluating GMM params...')
    _, gmm = cooc_utils.estimate_gmm(data_image, k=opt.kVal, sample_rate=cooc_utils.SAMPLE_RATE)
    print('\t Done.')

    try:
        cooc_vol = np.load(results_folder + 'cooc_vol.npy')
    except:
        print("Didn't find Cooc Volume, so building one..")
        cooc_vol = cooc_utils.cooc_finder(data_image, im_gmm=gmm, f_cooc=f_gauss,
                                          k=opt.kVal, requires_tensor=False)
        np.save(results_folder + 'cooc_vol', cooc_vol) # To make subsequent runs on the same image to be faster
    try:
        train_top_idx = np.load(results_folder + 'top_idx.npy')
        train_left_idx = np.load(results_folder + 'left_idx.npy')
    except:
        print("Didn't find train set, using entire space for crops..")
        train_top_idx, train_left_idx = [], []
    image_name = sample_image_filename.split(os.sep)[-1].split('.')[0]  # Extracting the image name

    return network_trained, data_image, cooc_vol, image_name, gmm, f_gauss, train_top_idx, train_left_idx


def main_function():
    global TRAIN_TOP, TRAIN_LEFT
    netG, data_image, cooc_vol, image_name, im_gmm, f_cooc, TRAIN_TOP, TRAIN_LEFT = get_folder_details()
    spatial_size = opt.imageSize // 2 ** nDep
    latent_size = opt.kVal ** 2
    dir_to_save = opt.outputFolder
    scale_factor_cooc = spatial_size / opt.imageSize
    noise_in = setNoise(torch.FloatTensor(1, latent_size, spatial_size, spatial_size).to(DEVICE))

    if opt.evalFunc == 'f_d':
        print('Doing Fidelity and Diversity Evaluation..')
        fidelity_diversity(data_image, cooc_vol, latent_size, spatial_size, opt.imageSize, image_name, dir_to_save,
                           netG, number_of_samples=8)
    elif opt.evalFunc == 'interp':
        print('Doing Interpolation Evaluation..')
        cooc_interpolation_and_extrap(data_image, cooc_vol, scale_factor_cooc, noise_in, opt.imageSize, image_name,
                                      dir_to_save,
                                      netG, number_of_samples=5, extra=2)
    elif opt.evalFunc == 'write_tex':
        print('Doing Writing Using Textures Evaluation..')
        written_texture(data_image, cooc_vol, scale_factor_cooc, opt.imageSize, image_name, dir_to_save, netG)
    else:
        print("Please input the test to do!")


COLOR_ARRAY = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [0, 0, 0],
               [255, 255, 255]]

if __name__ == '__main__':
    main_function()
