import nibabel as nib
import numpy as np
import random
from scipy.ndimage import rotate
import os
from tqdm import tqdm
import pydicom
import os
import random
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import label
import scipy
from scipy.spatial import distance
from tqdm import tqdm
#from datetime import datetime
from pathlib import Path
import argparse

# do not change
ORIGINAL_DICOM_IMAGE_PATH = './SCD_IMAGES_05/SCD0004001/CINELAX_301/IM-0003-0001.dcm'
ORIGINAL_FOLDER = './SCOTHEART_inference_output'
CARDIAC_FOLDER = './SCOTHEART_cardiac_coordinate'

def relabel_mask_and_apply_rotation(data_path : Path, output_path : Path):

    data_list = os.listdir(data_path)

    for i in tqdm(range(len(data_list))):
        each_path = os.path.join(data_path, data_list[i])
        each_nifti = nib.load(each_path)
        tmp_arr = each_nifti.get_fdata()
        tmp_aff = each_nifti.get_qform()

        tmp_arr_relabel = np.zeros(tmp_arr.shape)
        #tmp_arr_relabel[tmp_arr == 1] = 2
        #tmp_arr_relabel[tmp_arr == 2] = 1
        #tmp_arr_relabel[tmp_arr == 3] = 3
        #tmp_arr_relabel[tmp_arr == 5] = 5
        #tmp_arr_relabel[tmp_arr == 6] = 4

        tmp_arr_relabel[tmp_arr == 1] = 2
        tmp_arr_relabel[tmp_arr == 2] = 1
        tmp_arr_relabel[tmp_arr == 3] = 3
        tmp_arr_relabel[tmp_arr == 4] = 5
        tmp_arr_relabel[tmp_arr == 5] = 4

        # 10% probability
        if random.randint(-1, 8) < 0:
            # angle in degree
            angle = random.randint(-5, 5)
            tmp_arr_new = rotate(tmp_arr_relabel, angle, axes= (1, 2), reshape= False, order = 0, mode="constant")
            output_label_nifti = nib.Nifti1Image(tmp_arr_new, affine=tmp_aff)
        else:
            output_label_nifti = nib.Nifti1Image(tmp_arr_relabel, affine=tmp_aff)

        nib.save(output_label_nifti, os.path.join(output_path, data_list[i]))


########################################################################
# synthetic DICOM generation
########################################################################

# getting the largest connected component
def get_largest_cc(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

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



def write_nifti(meta_data, dicom_array, output_file):
    ##CONVERT TO NIFTI
    F11, F21, F31 = meta_data.ImageOrientationPatient[3:]
    F12, F22, F32 = meta_data.ImageOrientationPatient[:3]

    step = -np.cross(meta_data.ImageOrientationPatient[3:], meta_data.ImageOrientationPatient[:3]) * meta_data.SliceThickness

    delta_r, delta_c = meta_data.PixelSpacing
    Sx, Sy, Sz = meta_data.ImagePositionPatient

    affine = np.array(
        [
            [-F11 * delta_r, -F12 * delta_c, -step[0], -Sx],
            [-F21 * delta_r, -F22 * delta_c, -step[1], -Sy],
            [F31 * delta_r, F32 * delta_c, step[2], Sz],
            [0, 0, 0, 1]
        ]
    )

    new_nifti = np.zeros((dicom_array.shape[0], dicom_array.shape[1], 1))
    new_nifti[:,:,0] = dicom_array
    
    img_nii = nib.Nifti1Image(new_nifti.astype(np.uint8), affine)
    nib.save(img_nii, output_file)


def generate_nifti_from_dense_segmentation(path, output_folder, rotation_main_path):

    """
    generate dicom file from dense label map

    :path (str): the path of dense label map
    :motion option (boolean): if Ture, we shift array with given pixel and direction. otherwise, no motion data
    :return: multiple dicom files (2CH LAX, 4CH LAX and 9 SAX)
    """

    rotation_data_path = os.path.join(rotation_main_path, os.path.basename(path))
    dense_label_rotated = nib.load(rotation_data_path).get_fdata()

    dense_label = nib.load(path).get_fdata()
    seg_LV_tmp = (dense_label == 2).astype(dense_label.dtype)
    # clear unwanted segmentation (find the largest connected component)
    seg_LV = get_largest_cc(seg_LV_tmp).astype(dense_label.dtype)

    seg_LA_tmp = (dense_label == 5).astype(dense_label.dtype)
    seg_LA = get_largest_cc(seg_LA_tmp).astype(dense_label.dtype)

    seg_RV_tmp = (dense_label == 3).astype(dense_label.dtype)
    seg_RV = get_largest_cc(seg_RV_tmp).astype(dense_label.dtype)

    seg_RA_tmp = (dense_label == 4).astype(dense_label.dtype)
    seg_RA = get_largest_cc(seg_RA_tmp).astype(dense_label.dtype)

    seg_binary_tmp = (dense_label != 0).astype(dense_label.dtype)
    seg_binary = get_largest_cc(seg_binary_tmp).astype(dense_label.dtype)

    # mitral valve center
    seg_LA_dilation = scipy.ndimage.binary_dilation(seg_LA).astype(dense_label.dtype)
    mvc = np.mean(np.where(np.multiply(seg_LV, seg_LA_dilation) == 1), axis=1)

    seg_RA_dilation = scipy.ndimage.binary_dilation(seg_RA).astype(dense_label.dtype)
    tvc = np.mean(np.where(np.multiply(seg_RV, seg_RA_dilation) == 1), axis=1)

    # apex
    LV_points = np.where(seg_LV == 1)
    max_d = float('-inf')
    apex = None
    for i in range(len(LV_points[0])):
        if len(dense_label.shape) == 3:
            tmp_point = np.array([LV_points[0][i], LV_points[1][i], LV_points[2][i]])
        else:
            tmp_point = np.array([LV_points[0][i], LV_points[1][i]])

        tmp_distance = distance.euclidean(mvc, tmp_point)
        if tmp_distance > max_d:
            max_d = tmp_distance
            apex = tmp_point

    # center of the heart
    coh = np.mean(np.where(seg_binary == 1), axis=1)

    # centroid's of LV and RV
    lvc = np.mean(np.where(seg_LV == 1), axis=1)
    rvc = np.mean(np.where(seg_RV == 1), axis=1)

    output_loc = os.path.join(output_folder, os.path.basename(path)[:6])
    if not os.path.exists(output_loc):
        os.makedirs(output_loc)
        os.makedirs(output_loc + '/motion_corrupted')
        os.makedirs(output_loc + '/motion_free')

    # copy dicom header info from real MRI data
    dataset = pydicom.dcmread(ORIGINAL_DICOM_IMAGE_PATH)

    # Interpolation function
    xi = np.arange(0, 160, 1)
    yj = np.arange(0, 160, 1)
    zk = np.arange(0, 160, 1)
    f = RegularGridInterpolator((xi, yj, zk), dense_label_rotated, method="nearest", bounds_error=False, fill_value=0)

    ###########################################
    # 4CH-LAX
    ############################################
    vec_one = apex - mvc
    vec_two_tmp = tvc - mvc
    vec_three = np.cross(vec_one, vec_two_tmp)
    vec_two = np.cross(vec_three, vec_one)

    iop_one = vec_one / np.linalg.norm(vec_one)
    iop_two = vec_two/ np.linalg.norm(vec_two)

    plane_normal = np.cross(iop_one, iop_two)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    vector = coh - mvc
    distance_vec = np.dot(vector, plane_normal)
    projected_point = coh - distance_vec * plane_normal
    ipp = projected_point

    # generate all possible points inside this plane based on new IOP and original IPP with unit spacing
    points = []
    for x in np.arange(-80, 80, 1):
        for y in np.arange(-80, 80, 1):
            tmp_p = ipp + x * 1 * iop_one + y * 1 * iop_two
            points.append(tmp_p)

    # Interpolate values based on learnt interpolator
    fp = f(points)
    # resize into a (160, 160) plane
    img_4ch = np.zeros((160, 160))
    for kx in range(160):
        for ky in range(160):
            img_4ch[kx, ky] = fp[kx * 160 + ky]

    dicom_array = dataset.pixel_array
    gt = img_4ch.T

    # motion_free and motion_corrupted data saving (no motion simulation for 4CH)
    shift_pixel_1 = 0
    shift_pixel_2 = 0
    direction = np.array([shift_pixel_1, shift_pixel_2])
    motion_data = apply_affine_transformation_data(gt, direction)
    dicom_array = np.short(motion_data)
    dataset.ImagePositionPatient = list(points[0])
    dataset.ImageOrientationPatient = list(iop_one) + list(iop_two)
    dataset.PixelData = dicom_array.tobytes()
    dataset.PixelSpacing = [1, 1]
    dataset.Rows, dataset.Columns = dicom_array.shape
    dataset.SliceThickness = 1
    dataset.SpacingBetweenSlices = 1

    # Save to nifti
    output_file = os.path.join(output_loc + '/motion_free', os.path.basename(path)[:6] + '_LAX_4ch_' + str(shift_pixel_1) + '_'
                                 + str(shift_pixel_2) + '.nii.gz')
    
    write_nifti(dataset, dicom_array, output_file)

    output_file = os.path.join(output_loc + '/motion_corrupted', os.path.basename(path)[:6] + '_LAX_4ch_' + str(shift_pixel_1) + '_'
                                 + str(shift_pixel_2) + '.nii.gz')
    
    write_nifti(dataset, dicom_array, output_file)

    ############################################
    # 3CH-LAX [-3,3]
    ############################################
    data_name = os.path.basename(path)
    original_data_name = '110002_CE_label_maps.nii.gz'#data_name.replace('_cardiac_coordinate_space_id_affine', '')
    cardiac_data_name = '110002_CE_label_maps_cardiac_coordinate_space.nii.gz'#data_name.replace('_id_affine', '')
    original_path = os.path.join(ORIGINAL_FOLDER, original_data_name)
    cardiac_path = os.path.join(CARDIAC_FOLDER, cardiac_data_name)

    original_data = nib.load(original_path).get_fdata()
    original_affine = nib.load(original_path).affine
    cardiac_affine = nib.load(cardiac_path).affine


    seg_LV_tmp_original = (original_data == 1).astype(original_data.dtype)
    seg_LV_original = get_largest_cc(seg_LV_tmp_original).astype(original_data.dtype)

    seg_aorta_tmp_original = (original_data == 6).astype(original_data.dtype)
    seg_aorta_original = get_largest_cc(seg_aorta_tmp_original).astype(original_data.dtype)

    seg_aorta_dilation = scipy.ndimage.binary_dilation(seg_aorta_original).astype(original_data.dtype)
    avc = np.mean(np.where(np.multiply(seg_LV_original, seg_aorta_dilation) == 1), axis=1)
    avc_cardiac = np.dot(np.linalg.inv(cardiac_affine), np.dot(np.append(avc, 1), original_affine))[:3]

    vec_one_tmp = apex - mvc
    vec_two = avc_cardiac - mvc
    vec_three = np.cross(vec_one_tmp, vec_two)
    vec_one = np.cross(vec_two, vec_three)

    iop_one = vec_one / np.linalg.norm(vec_one)
    iop_two = vec_two/ np.linalg.norm(vec_two)

    plane_normal = np.cross(iop_one, iop_two)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    vector = coh - mvc
    distance_vec = np.dot(vector, plane_normal)
    projected_point = coh - distance_vec * plane_normal
    ipp = projected_point

    theta = random.choice([-1, 1]) * (np.pi / random.randint(60, 180))
    # print(np.pi/theta)

    # rotation matrix R_x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta), np.cos(theta)]])
    # rotation matrix R_y
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])

    A_LAX = np.zeros((3, 3))
    third_col = np.cross(iop_one, iop_two)
    third_col = third_col/np.linalg.norm(third_col)
    A_LAX[:, 0] = iop_one
    A_LAX[:, 1] = iop_two
    A_LAX[:, 2] = third_col

    # 80% for rotation
    if random.randint(-1, 8) > 0:
        A_LAX = np.matmul(A_LAX, Rx)
    elif random.randint(-1, 8) > 0:
        A_LAX = np.matmul(A_LAX, Ry)

    # Corresponding new IOP
    iop_one = A_LAX[:, 0]
    iop_two = A_LAX[:, 1]

    # generate all possible points inside this plane based on new IOP and original IPP with unit spacing
    points = []
    for x in np.arange(-80, 80, 1):
        for y in np.arange(-80, 80, 1):
            tmp_p = ipp + x * 1 * iop_one + y * 1 * iop_two
            points.append(tmp_p)

    # Interpolate values based on learnt interpolator
    fp = f(points)
    # resize into a (160, 160) plane
    img_3ch = np.zeros((160, 160))
    for kx in range(160):
        for ky in range(160):
            img_3ch[kx, ky] = fp[kx * 160 + ky]

    dicom_array = dataset.pixel_array
    gt = img_3ch.T

    # motion_free data saving
    shift_pixel_1 = 0
    shift_pixel_2 = 0
    direction = np.array([shift_pixel_1, shift_pixel_2])
    motion_data = apply_affine_transformation_data(gt, direction)
    dicom_array = np.short(motion_data)
    dataset.ImagePositionPatient = list(points[0])
    dataset.ImageOrientationPatient = list(iop_one) + list(iop_two)
    dataset.PixelData = dicom_array.tobytes()
    dataset.PixelSpacing = [1, 1]
    dataset.Rows, dataset.Columns = dicom_array.shape
    dataset.SliceThickness = 1
    dataset.SpacingBetweenSlices = 1
    #dataset.save_as(os.path.join(output_loc + '/motion_free', os.path.basename(path)[:6] + '_LAX_3ch_' + str(shift_pixel_1) + '_'
    #                             + str(shift_pixel_2) + '.dcm'))

    output_file = os.path.join(output_loc + '/motion_free', os.path.basename(path)[:6] + '_LAX_3ch_' + str(shift_pixel_1) + '_'
                                 + str(shift_pixel_2) + '.nii.gz')
    write_nifti(dataset, dicom_array, output_file)


    # motion_corrupted data saving
    shift_pixel_1 = int(np.random.normal(0, 3.5, 1)[0])
    shift_pixel_2 = int(np.random.normal(0, 3.5, 1)[0])
    direction = np.array([shift_pixel_1, shift_pixel_2])
    motion_data = apply_affine_transformation_data(gt, direction)
    dicom_array = np.short(motion_data)
    dataset.ImagePositionPatient = list(points[0])
    dataset.ImageOrientationPatient = list(iop_one) + list(iop_two)
    dataset.PixelData = dicom_array.tobytes()
    dataset.PixelSpacing = [1, 1]
    dataset.Rows, dataset.Columns = dicom_array.shape
    dataset.SliceThickness = 1
    dataset.SpacingBetweenSlices = 1
    output_file = os.path.join(output_loc + '/motion_corrupted', os.path.basename(path)[:6] + '_LAX_3ch_' + str(shift_pixel_1) + '_'
                                 + str(shift_pixel_2) + '.nii.gz')
    write_nifti(dataset, dicom_array, output_file)

    #############################################
    # 2CH-LAX, rotation degree range: [-3, 3]
    #############################################
    vec_one = apex - mvc
    vec_three = rvc - mvc
    vec_two = np.cross(vec_three, vec_one)

    iop_one = vec_one/np.linalg.norm(vec_one)
    iop_two = vec_two/np.linalg.norm(vec_two)

    plane_normal = np.cross(iop_one, iop_two)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    vector = coh - mvc
    distance_vec = np.dot(vector, plane_normal)
    projected_point = coh - distance_vec * plane_normal
    ipp = projected_point

    theta = np.random.choice([-1, 1]) * (np.pi / random.randint(60, 180))

    # rotation matrix R_x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta), np.cos(theta)]])
    # rotation matrix R_y
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])


    # original affine matrix for SAX
    A_LAX = np.zeros((3, 3))
    third_col = np.cross(iop_one, iop_two)
    third_col = third_col/ np.linalg.norm(third_col)
    A_LAX[:, 0] = iop_one
    A_LAX[:, 1] = iop_two
    A_LAX[:, 2] = third_col

    # 80% rotation
    if random.randint(-1, 8) > 0:
        A_LAX = np.matmul(A_LAX, Rx)
    elif random.randint(-1, 8) > 0:
        A_LAX = np.matmul(A_LAX, Ry)

    # Corresponding new IOP
    iop_one = A_LAX[:, 0]
    iop_two = A_LAX[:, 1]

    # generate all possible points inside this plane based on new IOP and original IPP with unit spacing
    points = []
    for x in np.arange(-80, 80, 1):
        for y in np.arange(-80, 80, 1):
            tmp_p = ipp + x * 1 * iop_one + y * 1 * iop_two
            points.append(tmp_p)

    # Interpolate values based on learnt interpolator
    fp = f(points)
    # resize into a (160, 160) plane
    img_2ch = np.zeros((160, 160))
    for kx in range(160):
        for ky in range(160):
            img_2ch[kx, ky] = fp[kx * 160 + ky]

    dicom_array = dataset.pixel_array
    gt = img_2ch.T

    # motion_free data saving
    shift_pixel_1 = 0
    shift_pixel_2 = 0
    direction = np.array([shift_pixel_1, shift_pixel_2])
    motion_data = apply_affine_transformation_data(gt, direction)
    dicom_array = np.short(motion_data)
    dataset.ImagePositionPatient = list(points[0])
    dataset.ImageOrientationPatient = list(iop_one) + list(iop_two)
    dataset.PixelData = dicom_array.tobytes()
    dataset.PixelSpacing = [1, 1]
    dataset.Rows, dataset.Columns = dicom_array.shape
    dataset.SliceThickness = 1
    dataset.SpacingBetweenSlices = 1
    #dataset.save_as(os.path.join(output_loc + '/motion_free', os.path.basename(path)[:6] + '_LAX_2ch_' + str(shift_pixel_1) + '_'
    #                             + str(shift_pixel_2) + '.dcm'))

    output_file = os.path.join(output_loc + '/motion_free', os.path.basename(path)[:6] + '_LAX_2ch_' + str(shift_pixel_1) + '_'
                                 + str(shift_pixel_2) + '.nii.gz')
    write_nifti(dataset, dicom_array, output_file)

    # motion_corrupted data saving
    shift_pixel_1 = int(np.random.normal(0, 3.5, 1)[0])
    shift_pixel_2 = int(np.random.normal(0, 3.5, 1)[0])
    direction = np.array([shift_pixel_1, shift_pixel_2])
    motion_data = apply_affine_transformation_data(gt, direction)
    dicom_array = np.short(motion_data)
    dataset.ImagePositionPatient = list(points[0])
    dataset.ImageOrientationPatient = list(iop_one) + list(iop_two)
    dataset.PixelData = dicom_array.tobytes()
    dataset.PixelSpacing = [1, 1]
    dataset.Rows, dataset.Columns = dicom_array.shape
    dataset.SliceThickness = 1
    dataset.SpacingBetweenSlices = 1
    #dataset.save_as(os.path.join(output_loc + '/motion_corrupted', os.path.basename(path)[:6] + '_LAX_2ch_' + str(shift_pixel_1) + '_'
    #                             + str(shift_pixel_2) + '.dcm'))
    output_file = os.path.join(output_loc + '/motion_corrupted', os.path.basename(path)[:6] + '_LAX_2ch_' + str(shift_pixel_1) + '_'
                                 + str(shift_pixel_2) + '.nii.gz')
    write_nifti(dataset, dicom_array, output_file)

    ##################################################
    # Stack of SAXs, rotate degree range: [-10, -3] & [3, 10]
    ##################################################
    basal_index = int(np.round(mvc[-1]))
    apex_index = int(np.round(apex[-1]))

    number_of_slices = random.randint(7, 11)
    max_length_lv = np.abs(basal_index - apex_index)
    slices_gap = int(np.round(max_length_lv / number_of_slices))

    theta = random.choice([-1, 1]) * (np.pi / random.randint(18, 60))

    # rotation matrix R_x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta), np.cos(theta)]])
    # rotation matrix R_y
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])

    # original affine matrix for SAX
    A_SAX = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])

    # 80% probability
    if random.randint(-1, 8) > 0:
        A_SAX = np.matmul(A_SAX, Rx)
    if random.randint(-1, 8) > 0:
        A_SAX = np.matmul(A_SAX, Ry)

    # Corresponding new IOP
    iop_one = A_SAX[:, 0]
    iop_two = A_SAX[:, 1]

    # basal and apex slice drop out (50%)
    drop_out_val_basal = random.randint(0,1)
    drop_out_val_apex = random.randint(0,1)

    print(os.path.basename(path)[:6] + ':' + str(number_of_slices+1))

    for i in range(number_of_slices+1):

        ipp = np.array([0, 0, min(apex_index, basal_index) + slices_gap * i])

        # generate all possible points inside this plane based on new IOP and original IPP with unit spacing
        points = []
        for x in np.arange(0, 160, 1):
            for y in np.arange(0, 160, 1):
                tmp_p = ipp + x * 1 * iop_one + y * 1 * iop_two
                points.append(tmp_p)

        # Interpolate values based on learnt interpolator
        fp = f(points)
        # resize into a (160, 160) plane
        img_sax = np.zeros((160, 160))
        for kx in range(160):
            for ky in range(160):
                img_sax[kx, ky] = fp[kx * 160 + ky]

        dicom_array = dataset.pixel_array
        gt = img_sax.T

        # motion_free data saving
        shift_pixel_1 = 0
        shift_pixel_2 = 0
        direction = np.array([shift_pixel_1, shift_pixel_2])
        motion_data = apply_affine_transformation_data(gt, direction)
        dicom_array = np.short(motion_data)
        dataset.ImagePositionPatient = list(ipp)
        dataset.ImageOrientationPatient = list(iop_one) + list(iop_two)
        dataset.PixelData = dicom_array.tobytes()
        dataset.PixelSpacing = [1, 1]
        dataset.Rows, dataset.Columns = dicom_array.shape
        dataset.SliceThickness = 1
        dataset.SpacingBetweenSlices = 1
        # random dropout the basal slice and apex slice
        if i == 0 and drop_out_val_apex:
            output_file = os.path.join(output_loc + '/motion_free', os.path.basename(path)[:6] + '_SAX_' + str(i + 1) + '_' + str(shift_pixel_1)
                             + '_' + str(shift_pixel_2) + '_apex.nii.gz')
            write_nifti(dataset, dicom_array, output_file)
        
        elif i == number_of_slices and drop_out_val_basal:
            output_file = os.path.join(output_loc + '/motion_free', os.path.basename(path)[:6] + '_SAX_' + str(i + 1) + '_' + str(shift_pixel_1)
                             + '_' + str(shift_pixel_2) + '_basal.nii.gz')
            write_nifti(dataset, dicom_array, output_file)
        elif 0 < i < number_of_slices and len(np.unique(motion_data)) != 1:
            output_file = os.path.join(output_loc + '/motion_free', os.path.basename(path)[:6] + '_SAX_' + str(i + 1) + '_' + str(shift_pixel_1)
                             + '_' + str(shift_pixel_2) + '.nii.gz')
            write_nifti(dataset, dicom_array, output_file)

        # motion_corrupted data saving
        shift_pixel_1 = int(np.random.normal(0, 3.5, 1)[0])
        shift_pixel_2 = int(np.random.normal(0, 3.5, 1)[0])
        direction = np.array([shift_pixel_1, shift_pixel_2])
        motion_data = apply_affine_transformation_data(gt, direction)
        dicom_array = np.short(motion_data)
        dataset.ImagePositionPatient = list(ipp)
        dataset.ImageOrientationPatient = list(iop_one) + list(iop_two)
        dataset.PixelData = dicom_array.tobytes()
        dataset.PixelSpacing = [1, 1]
        dataset.Rows, dataset.Columns = dicom_array.shape
        dataset.SliceThickness = 1
        dataset.SpacingBetweenSlices = 1
        # random dropout the basal slice and apex slice
        if i == 0 and drop_out_val_apex:
            output_file = os.path.join(output_loc + '/motion_corrupted', os.path.basename(path)[:6] + '_SAX_' + str(i + 1) + '_' + str(shift_pixel_1)
                             + '_' + str(shift_pixel_2) + '_apex.nii.gz')
            print(os.path.basename(path)[:6] + ':apex slices saved')
            write_nifti(dataset, dicom_array, output_file)

        elif i == number_of_slices and drop_out_val_basal:
            output_file = os.path.join(output_loc + '/motion_corrupted', os.path.basename(path)[:6] + '_SAX_' + str(i + 1) + '_' + str(shift_pixel_1)
                             + '_' + str(shift_pixel_2) + '_basal.nii.gz')
            print(os.path.basename(path)[:6] + ':basal slice saved')
            write_nifti(dataset, dicom_array, output_file)

        elif 0 < i < number_of_slices and len(np.unique(motion_data)) != 1:
            output_file = os.path.join(output_loc + '/motion_corrupted', os.path.basename(path)[:6] + '_SAX_' + str(i + 1) + '_' + str(shift_pixel_1)
                             + '_' + str(shift_pixel_2) + '.nii.gz')
            write_nifti(dataset, dicom_array, output_file)

    return


if __name__ == '__main__':

    # ######################################################################################
    parser = argparse.ArgumentParser(description='This script converts a 3D dense volume into a set of 2D segmentation')
    parser.add_argument('-i', '--input', type=Path,
                        help='Folder containing the 3D dense volume', default='./toy_data')
    parser.add_argument('-o', '--output_path', type=Path,
                        help='Path where to save the synthesized 2D segmentations', default='./synthesized_segmentations') 
    
    args = parser.parse_args()
    assert Path(args.input).exists(), \
        f'Cannot not find {args.input}!'
    
    args.output_path.mkdir(parents=True, exist_ok=True)

    output_rotation = args.output_path / 'data_after_rotation'
    output_rotation.mkdir(parents=True, exist_ok=True)
    relabel_mask_and_apply_rotation(data_path = args.input, output_path = output_rotation)

    data_list = sorted(os.listdir(args.input))
    error_case = []

    for k in tqdm(range(len(data_list))):
        path = os.path.join(args.input, data_list[k])
        generate_nifti_from_dense_segmentation(path, args.output_path, output_rotation)

