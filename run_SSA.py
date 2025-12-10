import numpy as np
import pydicom
import os
import nibabel as nib
import torch
import matplotlib as plt
plt.use('TkAgg')
import scipy
import json
from pathlib import Path
import argparse
from loguru import logger
from distutils.dir_util import copy_tree
import time 
from rich.progress import Progress
import re
from scipy.ndimage import map_coordinates

# Configuration
RV_LABEL = 3
RA_LABEL = 4
RV_MYO_LABEL = 6

LV_LABEL = 2
LA_LABEL = 5
LV_MYO_LABEL = 1

# @timing
def extract_info(path, dicom_format = True):
    """
    extract data array, affine matrix and slice thickness from dicom/nifti header information

    :path (str): the path of dense label map
    :dicom_format (boolean): whether the input file is dicom or not
    :return: a list of data array, affine matrix and slice thickness
    :data_array (numpy array): 2D array of labelmap slice
    :data_affine (numpy array ,4x4): affine matrix of the data
    :data_thickness (int): thickness of slices
    """
    if dicom_format:
        ds = pydicom.dcmread(path)
        data_array = ds.pixel_array.astype(np.float64)
        data_affine = np.eye(4)
        data_affine[:3, 0] = np.array(ds.ImageOrientationPatient[:3])
        data_affine[:3, 1] = np.array(ds.ImageOrientationPatient[3:])
        data_affine[:3, 2] = np.cross(data_affine[:3, 0], data_affine[:3, 1])
        # print(data_affine)
        data_affine[:3, 3] = np.array(ds.ImagePositionPatient)
        data_thickness = ds.SliceThickness

        return [data_array, data_affine, data_thickness]

    else:
        data = nib.load(path)
        data_array = data.get_fdata()
        data_affine = data.get_qform()
        data_thickness = data.header['pixdim'][3]

        return [data_array, data_affine, data_thickness]

# @timing
def intersection_resample(slice_a, slice_b):
    """
    calculate the intersection area by resampling slice_a (with thickness) into empty slice_b (same shape as slice_b)

    :slice_a (list): a list of data array, affine matrix and slice thickness for slice a
    :slice_b (list): a list of data array, affine matrix and slice thickness for slice b
    :return: a resampled intersection area in the empty slice_B
    :intersection_result (numpy array): 2D array of resampled intersection area (same shape as slice_b)
    """

    # read slice A and slice B information
    array_a = slice_a[0]
    affine_a = slice_a[1]
    #print(affine_a)
    #thickness_a = slice_a[2]
    thickness_a = 2
    array_b = slice_b[0]
    affine_b = slice_b[1]
    # thickness_b = 1
    thickness_b = slice_b[2]

    # create block A with given thickness
    a_with_thickness = np.zeros((int(thickness_a), array_a.shape[0], array_a.shape[1]))
    a_with_thickness[:, ...] = array_a
    # affine matrix for block A
    affine_a_with_thickness = np.zeros(affine_a.shape)
    affine_a_with_thickness[affine_a != 0] = affine_a[affine_a != 0]
    affine_a_with_thickness[:, -1] -= affine_a_with_thickness[:, 2] * (a_with_thickness.shape[0]//2)

    # coordinate transformation, ijk to xyz
    # https://medium.com/redbrick-ai/dicom-coordinate-systems-3d-dicom-for-computer-vision-engineers-pt-1-61341d87485f
    ij_index = (np.array(np.meshgrid(np.arange(0, array_b.shape[1]), np.arange(0, array_b.shape[0]), indexing='ij')).T.reshape(array_b.shape[0], array_b.shape[1], 2))
    ijk_index = np.zeros((array_b.shape[0], array_b.shape[1], 4))
    ijk_index[:, :, :2] = ij_index
    # for matrix multiplication, make the last row as all 1
    ijk_index[:, :, -1] = 1
    # each entry for matrix ijk_index is in (i, j, 0, 1) format
    ijk_index = ijk_index.T.reshape(4, array_b.shape[0] * array_b.shape[1])
    # affine matrix for intersection area, which will be used for defining grid later
    affine_inter = np.dot(np.linalg.inv(affine_a_with_thickness), affine_b)
    # xyz coordinate system
    xyz_index_temp = np.dot(affine_inter, ijk_index)
    # remove the last row (all 1)
    xyz_index = xyz_index_temp[:3, :].reshape(3, array_b.shape[1], array_b.shape[0]).T

    # input tensor [N, C, D_in, H_in, W_in] and grid tensor [N, D_out, H_out, W_out, 3] for the grid_sampling function
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    input_tensor = torch.from_numpy(a_with_thickness.reshape((1, 1, a_with_thickness.shape[0], a_with_thickness.shape[1], a_with_thickness.shape[2])))
    grid = torch.from_numpy(xyz_index.reshape((1, 1, array_b.shape[0], array_b.shape[1], 3))).type(torch.DoubleTensor)

    # normalization to [-1, 1] for grid tensor
    norm_factor_0 = (input_tensor.shape[2] - 1) / 2
    norm_factor_1 = (input_tensor.shape[3] - 1) / 2
    norm_factor_2 = (input_tensor.shape[4] - 1) / 2

    grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - norm_factor_2) / (norm_factor_2 + 0.5)
    grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - norm_factor_1) / (norm_factor_1 + 0.5)
    grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - norm_factor_0) / (norm_factor_0 + 0.5)

    # resampled intersection area [N, C, D_out, H_out, W_out]
    intersection_result = torch.nn.functional.grid_sample(input_tensor, grid, mode='nearest', padding_mode='zeros',
                                                          align_corners=False)[0, 0, 0, ...].numpy()

    return intersection_result

def cal_target(moving_img, template_img):
    """
    Elementwise label match count across unique labels (excluding background 0).
    """
    labels = np.unique(moving_img)
    labels = labels[labels != 0]  # exclude background label 0

    target = 0
    for label in labels:
        match = (moving_img == label) & (template_img == label)
        target += np.count_nonzero(match)

    return target

def cal_target_original(moving_img, template_img):
    """
    elementwise multiplication for each label and then sum it up as final target

    :moving_img (numpy array): labelmap of moving image
    :template_img (numpy array): image of combined intersection with other slices
    :return: sum of elementwise multiplication for each label
    """

    target = 0
    for i in list(np.unique(moving_img)):
        if i != 0:
            target += np.sum(np.multiply((moving_img == i), (template_img == i)))

    return target

def dice_score(moving_img, template_img):
    """
    dice score for each label and then sum it up as final target

    :moving_img (numpy array): labelmap of moving image
    :template_img (numpy array): image of combined intersection with other slices
    :return: sum of dice score for each label
    """

    dice = np.zeros(5)
    for i in list(np.unique(moving_img)):
        if i != 0:
            dice_label = 2*np.sum(np.multiply((moving_img == i), (template_img == i)))/(np.sum(moving_img == i) + np.sum(template_img == i))
            dice[int(i)-1] = dice_label

    dice = dice/len(np.unique(moving_img))

    return dice

# @timing
def grid_search(moving_img, template_img, lr_unit, ud_unit):
    """
    find the best direction for maximizing the overlapping area between original slices (LAX or SAX) and intersection imgae (all intersection area with other slices)

    :moving_img (numpy array): labelmap of moving image
    :template_img (numpy array): image of combined intersection with other slices
    :lr_unit (int): searching range for left and right direction, if lr_unit = 9, it means we are searching from [-4(most left), 4 (most right)]
    :ud_unit (int): searching range for up and down direction, if ud_unit = 9, it means we are searching from [-4(downmost), 4 (upmost)]
    :return: the best in-plane shift for maximizing the overlapping area
    :best_direction (tuple): (direction, corresponding target value)
    """

    # define a grid for searching directions, where i>0 (right), i<0 (left), j>0 (up) and j<0 (down)
    direction = np.array(np.meshgrid(np.arange(0, lr_unit), np.arange(0, ud_unit))).T
    direction = direction - direction.shape[0] // 2
    # total number of searching times
    total_num_search = direction.shape[0] * direction.shape[1]
    direction = direction.reshape(total_num_search, 2)

    # a list of target value (element-wise multiplication in my case) for later optimization (maximization in my case)
    target_val_list = []

    for d in range(total_num_search):
        corrected_img = np.zeros(moving_img.shape)

        # right
        if direction[d][0] > 0:
            # up
            if direction[d][1] > 0:
                corrected_img[0: moving_img.shape[0] - np.abs(direction[d][1]),
                np.abs(direction[d][0]): moving_img.shape[1]] = moving_img[np.abs(direction[d][1]):moving_img.shape[0],
                                                               0: moving_img.shape[1] - np.abs(direction[d][0])]
                temp_target = cal_target_original(corrected_img, template_img)
                #t = time.time()
                #temp_target = cal_target(corrected_img, template_img)
                #print(f"{time.time() - t} new {temp_target}")
                # temp_target = dice_score(corrected_img, template_img)
                # make sure the shift doesn't crop the image
                if np.sum(corrected_img > 0) == np.sum(moving_img > 0):
                    target_val_list.append((direction[d], temp_target))
            # down
            else:
                corrected_img[np.abs(direction[d][1]):moving_img.shape[0],
                np.abs(direction[d][0]): moving_img.shape[1]] = moving_img[0: moving_img.shape[0] - np.abs(direction[d][1]),
                                                               0: moving_img.shape[1] - np.abs(direction[d][0])]
                temp_target = cal_target(corrected_img, template_img)
                # temp_target = dice_score(corrected_img, template_img)
                if np.sum(corrected_img > 0) == np.sum(moving_img > 0):
                    target_val_list.append((direction[d], temp_target))
        # left
        else:
            # up
            if direction[d][1] > 0:
                corrected_img[0: moving_img.shape[0] - np.abs(direction[d][1]),
                0: moving_img.shape[1] - np.abs(direction[d][0])] = moving_img[np.abs(direction[d][1]):moving_img.shape[0],
                                                                   np.abs(direction[d][0]): moving_img.shape[1]]
                temp_target = cal_target(corrected_img, template_img)
                # temp_target = dice_score(corrected_img, template_img)
                if np.sum(corrected_img > 0) == np.sum(moving_img > 0):
                    target_val_list.append((direction[d], temp_target))
            # down
            else:
                corrected_img[np.abs(direction[d][1]):moving_img.shape[0],
                0: moving_img.shape[1] - np.abs(direction[d][0])] = moving_img[
                                                                   0: moving_img.shape[0] - np.abs(direction[d][1]),
                                                                   np.abs(direction[d][0]): moving_img.shape[1]]
                temp_target = cal_target(corrected_img, template_img)
                # temp_target = dice_score(corrected_img, template_img)
                if np.sum(corrected_img > 0) == np.sum(moving_img > 0):
                    target_val_list.append((direction[d], temp_target))

    best_direction = max(target_val_list, key=lambda x: x[1])

    return best_direction

def apply_affine_transformation(path_data, dicom_file_name, direction, dicom_format = True):
    """
    apply in-plane transformation (i.e., slice shifting) to the current slice

    :path_data (str): location of current data
    :dicom_file_name (str): name of current moving image/slice of a specific data
    :direction (tuple): [left/right, up/down]
    :iter (int): iteration number
    :return: new dicom file with same header information as before but updated content
    """
    if dicom_format:
        path_dicom = os.path.join(path_data, dicom_file_name)
        dataset = pydicom.dcmread(path_dicom)
        moving_img = dataset.pixel_array.astype(np.float64)
        corrected_img = np.zeros(moving_img.shape)

        # slice shifting by changing the content for the slice
        # right
        if direction[0] > 0:
            # up
            if direction[1] > 0:
                corrected_img[0: moving_img.shape[0] - np.abs(direction[1]),
                np.abs(direction[0]): moving_img.shape[1]] = moving_img[np.abs(direction[1]):moving_img.shape[0],
                                                             0: moving_img.shape[1] - np.abs(direction[0])]

            # down
            else:
                corrected_img[np.abs(direction[1]):moving_img.shape[0],
                np.abs(direction[0]): moving_img.shape[1]] = moving_img[0: moving_img.shape[0] - np.abs(direction[1]),
                                                             0: moving_img.shape[1] - np.abs(direction[0])]

        # left
        else:
            # up
            if direction[1] > 0:
                corrected_img[0: moving_img.shape[0] - np.abs(direction[1]),
                0: moving_img.shape[1] - np.abs(direction[0])] = moving_img[np.abs(direction[1]):moving_img.shape[0],
                                                                 np.abs(direction[0]): moving_img.shape[1]]

            # down
            else:
                corrected_img[np.abs(direction[1]):moving_img.shape[0],
                0: moving_img.shape[1] - np.abs(direction[0])] = moving_img[
                                                                 0: moving_img.shape[0] - np.abs(direction[1]),
                                                                 np.abs(direction[0]): moving_img.shape[1]]

        # save new dicom file with same header information as before
        moving_img = np.short(corrected_img)
        dataset.PixelData = moving_img.tobytes()
        dataset.save_as(path_dicom)
    else:
        path_dicom = os.path.join(path_data, dicom_file_name)
        data = nib.load(path_dicom)
        moving_img =data.get_fdata()[..., 0].T
        corrected_img = np.zeros(moving_img.shape)

        # slice shifting by changing the content for the slice
        # right
        if direction[0] > 0:
            # up
            if direction[1] > 0:
                corrected_img[0: moving_img.shape[0] - np.abs(direction[1]),
                np.abs(direction[0]): moving_img.shape[1]] = moving_img[np.abs(direction[1]):moving_img.shape[0],
                                                             0: moving_img.shape[1] - np.abs(direction[0])]

            # down
            else:
                corrected_img[np.abs(direction[1]):moving_img.shape[0],
                np.abs(direction[0]): moving_img.shape[1]] = moving_img[0: moving_img.shape[0] - np.abs(direction[1]),
                                                             0: moving_img.shape[1] - np.abs(direction[0])]

        # left
        else:
            # up
            if direction[1] > 0:
                corrected_img[0: moving_img.shape[0] - np.abs(direction[1]),
                0: moving_img.shape[1] - np.abs(direction[0])] = moving_img[np.abs(direction[1]):moving_img.shape[0],
                                                                 np.abs(direction[0]): moving_img.shape[1]]

            # down
            else:
                corrected_img[np.abs(direction[1]):moving_img.shape[0],
                0: moving_img.shape[1] - np.abs(direction[0])] = moving_img[
                                                                 0: moving_img.shape[0] - np.abs(direction[1]),
                                                                 np.abs(direction[0]): moving_img.shape[1]]

        # save new dicom file with same header information as before
        moving_img_new = corrected_img.T
        new_nifti = nib.Nifti1Image(moving_img_new[:, :, np.newaxis], data.affine)
        nib.save(new_nifti, path_dicom)

    return

def apply_affine_transformation_data(original_data, direction):
    """
    apply in-plane transformation (i.e., slice shifting) to the current slice

    :original_data (numpy array): array of current data
    :direction (tuple): (direction, corresponding target value)
    :return: new numpy array for the current slices
    """

    moving_img = original_data
    corrected_img = np.zeros(moving_img.shape)

    # slice shifting by changing the content for the slice
    # right
    if direction[0] > 0:
        # up
        if direction[1] > 0:
            corrected_img[0: moving_img.shape[0] - np.abs(direction[1]),
            np.abs(direction[0]): moving_img.shape[1]] = moving_img[np.abs(direction[1]):moving_img.shape[0],
                                                            0: moving_img.shape[1] - np.abs(direction[0])]

        # down
        else:
            corrected_img[np.abs(direction[1]):moving_img.shape[0],
            np.abs(direction[0]): moving_img.shape[1]] = moving_img[0: moving_img.shape[0] - np.abs(direction[1]),
                                                            0: moving_img.shape[1] - np.abs(direction[0])]

    # left
    else:
        # up
        if direction[1] > 0:
            corrected_img[0: moving_img.shape[0] - np.abs(direction[1]),
            0: moving_img.shape[1] - np.abs(direction[0])] = moving_img[np.abs(direction[1]):moving_img.shape[0],
                                                                np.abs(direction[0]): moving_img.shape[1]]

        # down
        else:
            corrected_img[np.abs(direction[1]):moving_img.shape[0],
            0: moving_img.shape[1] - np.abs(direction[0])] = moving_img[
                                                                0: moving_img.shape[0] - np.abs(direction[1]),
                                                                np.abs(direction[0]): moving_img.shape[1]]

    return corrected_img

#def find_max_and_indices_orignal(arr):
#    # Initialize variables
#    max_value = float('-inf')
#    max_indices = []
#
#    # Traverse the 2D array to find the maximum value and its indices
#    first_dim_half = int(arr.shape[0] // 2)
#    second_dim_half = int(arr.shape[1] // 2)
#    for i in range(first_dim_half-10, first_dim_half+10):
#        for j in range(second_dim_half-10, second_dim_half+10):
#            if arr[i][j] - max_value > 1e-5:
#                max_value = arr[i][j]
#                max_indices = [(i, j)]  # Reset indices list when a new max is found
#            elif abs(arr[i][j] - max_value) < 1e-5:
#                max_indices.append((i, j))  # Add index to list if it matches the current max
#
#    return max_value, max_indices

def find_max_and_indices(arr):
    first_dim_half = arr.shape[0] // 2
    second_dim_half = arr.shape[1] // 2

    # Define the slice range
    i_start = first_dim_half - 10
    i_end = first_dim_half + 10
    j_start = second_dim_half - 10
    j_end = second_dim_half + 10

    # Extract the subarray
    sub_arr = arr[i_start:i_end, j_start:j_end]

    # Find the maximum value
    max_value = np.max(sub_arr)

    # Get all indices of the maximum value in the subarray
    rel_indices = np.argwhere(np.abs(sub_arr - max_value) < 1e-5)

    # Convert subarray-relative indices to original array indices
    max_indices = [(i + i_start, j + j_start) for i, j in rel_indices]

    max_indices = [(int(x), int(y)) for x, y in max_indices]

    return max_value, max_indices

def fft_search(template_img, moving_img):
    fft_new = None
    label_list = np.unique(moving_img)

    for l in range(1, len(label_list)):
        label = label_list[l]
        moving_binary = (moving_img == label).astype(np.int32)
        template_binary = (template_img == label).astype(np.int32)
        tmp_fft = scipy.signal.correlate(template_binary, moving_binary, method='fft')
        if fft_new is None:
            fft_new = tmp_fft
        else:
            fft_new += tmp_fft

    fft_ref = scipy.signal.correlate(template_img, template_img, method='fft')

    peak_index_ref = find_max_and_indices(fft_ref)[1]
    peak_index = find_max_and_indices(fft_new)[1]

    max_shift_abs = 20
    best_direction = (0, 0)
    ref_index = peak_index_ref[0]  # use the first peak index as reference

    for current_index in peak_index:
        tmp_dir = (current_index[1] - ref_index[1], ref_index[0] - current_index[0])
        abs_diff = abs(tmp_dir[0]) + abs(tmp_dir[1])
        if abs_diff == 0:
            return tmp_dir
        elif abs_diff < max_shift_abs:
            max_shift_abs = abs_diff
            best_direction = tmp_dir

    return best_direction

def count_values(arr):
   unique, counts = np.unique(arr, return_counts=True)
   return dict(zip(unique, counts))


def calculate_shift(path_data: Path, case_name : str):

    dicom_list = sorted(os.listdir(path_data))
    path_dicom = path_data
    num_iter = 5
    #deleted_list = {}
    ssa_hist = {}

    #count = 0

    #my_count = 0
    for key in dicom_list:
        ssa_hist[key] = [0, 0]
    for j in range(num_iter):
        logger.info('----------------------------------------------------------')
        # iterate over dicom file under each data, start with LAX-4CH
        for k in range(len(dicom_list)):
            moving_img_name = dicom_list[k]
            # remove current slice for later intersection calculation (will add back later)
            dicom_list.remove(moving_img_name)
            moving_img_list = extract_info(os.path.join(path_dicom, moving_img_name), dicom_format=False)
            moving_img_list[0] = moving_img_list[0][...,0].T
            template_img = np.zeros(moving_img_list[0].shape)
            # we need delete the slice (like apex) which contains less information
            if len(np.unique(moving_img_list[0])) != 1:
                for s in range(len(dicom_list)):
                    tmp_slice_name = dicom_list[s]
                    tmp_slice_list = extract_info(os.path.join(path_dicom, tmp_slice_name), dicom_format=False)
                    tmp_slice_list[0] = tmp_slice_list[0][...,0].T
                    single_intersection = intersection_resample(tmp_slice_list, moving_img_list)
                    template_img = np.maximum(template_img, single_intersection)

                try:
                    ratio = count_values(template_img)[1.0] / count_values(template_img)[2.0]
                    num_ones = count_values(template_img)[1.0]
                except:
                    logger.info('This slice does not cut through LV :' + os.path.join(path_dicom, moving_img_name))
                    ratio = 0
                    num_ones = 0

                if (('sa' in moving_img_name.lower()) and (len(np.unique(template_img)) == 6) and (len(np.unique(moving_img_list[0])) < 3)) or (
                        ('sa' in moving_img_name.lower()) and (ratio < 0.2) and (num_ones < 5)):
                    moving_img_list_updated = extract_info(os.path.join(path_dicom, moving_img_name), dicom_format=False)
                    moving_img_list_updated[0] = np.zeros(moving_img_list_updated[0].shape)
                    new_nifti = nib.Nifti1Image(moving_img_list_updated[0], moving_img_list_updated[1])

                    nib.save(new_nifti, os.path.join(path_dicom, moving_img_name))
                    #count += 1
                    logger.error('deleted slice location:' + os.path.join(path_dicom, moving_img_name))
                else:
                    in_plane_shift = fft_search(template_img, moving_img_list[0])
                    #ssa_hist.setdefault(os.path.basename(path_dicom)+ '_' + moving_img_name, []).append((j+1, in_plane_shift))

                    ssa_hist[moving_img_name][0] += in_plane_shift[0]
                    ssa_hist[moving_img_name][1] += in_plane_shift[1]
                    logger.info('iteration:' + str(j+1) + '; slice name:' + moving_img_name + '; current shift:' + str(in_plane_shift))
                    # update dicom file
                    apply_affine_transformation(path_dicom, moving_img_name, in_plane_shift, False)
            # add back the slice name

            dicom_list.append(moving_img_name)


        logger.info('iteration' + str(j+1) + ' finished')

    return ssa_hist

def apply_shift(path_data: Path, json_path: Path):
    with open(json_path, 'r') as file:
        ssa_shift = json.load(file)

    key_set = set(ssa_shift.keys())  # faster lookups
    dicom_list_full = sorted(os.listdir(path_data))  # don't modify this list

    logger.info('----------------------------------------------------------')
    for moving_img_name in dicom_list_full:
        moving_img_path = os.path.join(path_data, moving_img_name)
        moving_img_list = extract_info(moving_img_path, dicom_format=False)
        moving_img_list[0] = moving_img_list[0][..., 0].T

        # Replace frame number with f1 (reference frame)
        ref_slice_name = re.sub(r'f\d+', 'f1', moving_img_name)

        if ref_slice_name in key_set:
            in_plane_shift = ssa_shift[ref_slice_name]
            apply_affine_transformation(path_data, moving_img_name, in_plane_shift, False)
        else:
            # Replace image with zero array
            moving_img_list_updated = extract_info(moving_img_path, dicom_format=False)
            moving_img_list_updated[0] = np.zeros_like(moving_img_list_updated[0])
            new_nifti = nib.Nifti1Image(moving_img_list_updated[0], moving_img_list_updated[1])
            nib.save(new_nifti, moving_img_path)
            logger.info(f"deleted slices: {os.path.basename(path_data)}_{moving_img_name}")

#def remove_rv_in_la(folder, output_folder):
#    # Load all segmentations once
#    seg_files = {f: nib.load(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith((".nii", ".nii.gz"))}
#
#    # Identify 4Ch file
#    fourch_file = next((f for f in seg_files if '4Ch' in f or '4CH' in f), None)
#    if not fourch_file:
#        raise RuntimeError("4Ch segmentation not found.")
#
#    fourch_img = seg_files[fourch_file]
#    fourch_data = fourch_img.get_fdata()
#    affine_4ch = fourch_img.affine
#    inv_affine_4ch = np.linalg.inv(affine_4ch)
#
#    # Labels constants for clarity
#    #RV_LABELS = {RV_LABEL, RV_MYO_LABEL}
#    #LV_LABELS = {LV_LABEL, LV_MYO_LABEL}
#
#    for fname, sax_img in seg_files.items():
#        if 'SA' not in fname or fname == fourch_file:
#            continue
#
#        sax_data = sax_img.get_fdata()
#        affine_sax = sax_img.affine
#
#        corrected_data = sax_data.copy()
#
#        # Find all RV voxels in SAX
#        rv_voxels = np.argwhere(sax_data == RV_LABEL)
#        if rv_voxels.size > 0:
#            rv_voxels_h = np.c_[rv_voxels, np.ones(len(rv_voxels))]
#            world_coords_rv = (affine_sax @ rv_voxels_h.T).T[:, :3]
#            vox_4ch_rv = (inv_affine_4ch @ np.c_[world_coords_rv, np.ones(len(world_coords_rv))].T).T[:, :3].T
#            sampled_labels_rv = map_coordinates(fourch_data, vox_4ch_rv, order=0, mode='nearest')
#            intersecting_indices_ra = sampled_labels_rv == RA_LABEL
#            num_overlap_ra = np.sum(intersecting_indices_ra)
#        else:
#            num_overlap_ra = 0
#
#        # Find all LV voxels in SAX
#        lv_voxels = np.argwhere(sax_data == LV_LABEL)
#        if lv_voxels.size > 0:
#            lv_voxels_h = np.c_[lv_voxels, np.ones(len(lv_voxels))]
#            world_coords_lv = (affine_sax @ lv_voxels_h.T).T[:, :3]
#            vox_4ch_lv = (inv_affine_4ch @ np.c_[world_coords_lv, np.ones(len(world_coords_lv))].T).T[:, :3].T
#            sampled_labels_lv = map_coordinates(fourch_data, vox_4ch_lv, order=0, mode='nearest')
#            intersecting_indices_la = sampled_labels_lv == LA_LABEL
#            intersecting_indices_rv = sampled_labels_lv == RV_LABEL
#            num_overlap_la = np.sum(intersecting_indices_la)
#            num_overlap_rv = np.sum(intersecting_indices_rv)
#        else:
#            num_overlap_la = 0
#            num_overlap_rv = 0
#
#        if num_overlap_ra > 0:
#            corrected_data[corrected_data == RV_LABEL] = 0
#            corrected_data[corrected_data == RV_MYO_LABEL] = 0
#
#        if num_overlap_la > 0:
#            corrected_data[corrected_data == LV_LABEL] = 0
#            corrected_data[corrected_data == LV_MYO_LABEL] = 0
#
#        if num_overlap_rv > 0:
#            corrected_data[corrected_data == LV_LABEL] = 0
#            corrected_data[corrected_data == LV_MYO_LABEL] = 0
#            corrected_data[corrected_data == RV_LABEL] = 0
#            corrected_data[corrected_data == RV_MYO_LABEL] = 0
#
#        if (num_overlap_ra + num_overlap_la + num_overlap_rv) == 0:
#            continue
#
#        corrected_img = nib.Nifti1Image(corrected_data, sax_img.affine, sax_img.header)
#        corrected_path = os.path.join(output_folder, fname)
#        nib.save(corrected_img, corrected_path)

if __name__ == '__main__':

    # ######################################################################################
    parser = argparse.ArgumentParser(description='This script performs breath-hold misregistration correction by calculating the corresponding shift from the ED frame')
    parser.add_argument('-i', '--input', type=Path,
                        help='Folder containing the niftis at ED', default='./example_data')
    parser.add_argument('-o', '--output_path', type=Path,
                        help='Path where to save the applied translations json file and corrected nifitis', default='./corrected_niftis')
    parser.add_argument('-ed', '--ed_frame', type=int,
                        help='ED frame number', default=1)
    parser.add_argument('-s', '--step', type=str,
                        help='Calculate shifts or apply shifts: calculate shfits to apply on the ED frame. Infer: apply calculated shifts to remaining frames', default='calculate')   
    parser.add_argument("--log", action="store_true", help="Enable logging to a file and console.")
    
    args = parser.parse_args()

    assert Path(args.input).exists(), \
        f'Cannot not find {args.input}!'

    args.output_path.mkdir(parents=True, exist_ok=True)

    assert args.step in ['calculate', 'infer'], \
        f'-s must be "calculate" or "infer". {args.step} given.'

    # set list of cases to process
    case_list = os.listdir(args.input)
    case_dirs = [Path(args.input, Path(case).name) for case in case_list]

    log_level = "DEBUG"
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
    logger.info(f"Found {len(case_list)} cases to process.")

    if not args.log:
        logger.remove()

    start_time = time.time()

    for case in case_dirs:
        logger.info(f"Processing {os.path.basename(case)}")
        output_nifti = args.output_path / case.name
        Path(output_nifti).mkdir(parents=True, exist_ok=True)

        if args.step == 'calculate':
            copy_tree(args.input / case.name / f'{case.name}_{args.ed_frame}', output_nifti / f'{case.name}_{args.ed_frame}')
            shifts = calculate_shift(output_nifti / f'{case.name}_{args.ed_frame}', case.name)
            #remove_rv_in_la(output_nifti / f'{case.name}_{args.ed_frame}', output_nifti / f'{case.name}_{args.ed_frame}')
            with open(os.path.join(output_nifti, f'{case.name}_translation_file.json'), 'w') as f:
                json.dump(shifts, f)

        if args.step == 'infer':
            json_file = output_nifti / f'{case.name}_translation_file.json'
            assert json_file.exists(), \
                f'Cannot find shifts file. {json_file} does not exist.'
            
            time_frame = [name for name in os.listdir(Path(args.input / case.name)) ]

            frame_name = [int(re.search(f'{case.name}_*(\d+)', str(file), re.IGNORECASE)[1]) for file in time_frame]
            frame_name = sorted(frame_name)

            with Progress(transient=True) as progress:
                task = progress.add_task(f"Processing {len(frame_name)-1} frames", total=len(frame_name)-1)
                console = progress

                for frame in frame_name:
                    if frame != args.ed_frame:
                        copy_tree(args.input / case.name / f'{case.name}_{frame}', output_nifti / f'{case.name}_{frame}')
                        apply_shift(output_nifti / f'{case.name}_{frame}', json_file)
                        #remove_rv_in_la(output_nifti / f'{case.name}_{frame}', output_nifti / f'{case.name}_{frame}')

                        progress.advance(task)

    logger.info(f"Total cases processed: {len(case_dirs)}")
    logger.info(f"Total time: {time.time() - start_time}")
    logger.success(f'Done. Results are saved in {args.output_path}')