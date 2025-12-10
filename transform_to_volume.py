##########################################################################################
# resample all dicom files into a 3D sparse volume for UKBB data (grid-sampling version)
##########################################################################################
from run_SSA import extract_info
import nibabel as nib
import numpy as np
import os
import torch
import scipy
from scipy.spatial import distance
from pathlib import Path
import argparse
from loguru import logger
import time
from scipy.ndimage import label
import torch

def getLargestCC(segmentation):
    # Label connected components
    labels, num_labels = label(segmentation)
    
    if num_labels == 0:
        raise ValueError("No connected components found in the segmentation.")
    
    # Compute the size of each connected component using np.bincount on labels
    component_sizes = np.bincount(labels.ravel())
    
    # Find the largest component index, excluding the background label (0)
    largest_component = np.argmax(component_sizes[1:]) + 1  # Exclude background (0)
    
    # Create a binary mask for the largest connected component
    largestCC = labels == largest_component
    
    return largestCC

def landmark_gen(path, output_size = 160):
    """
    find the landmarks (apex, mvc, tvc, lvc, rvc, coh) for the cine-MRI and MRA data

    :path (str): location of cine_MRI and MRA (cine-MRI's array should be three dimension with the last dimension as 1, i.e., 2D slices)
    :output_size (int): the shape of resampled data, we assume the output should be isotropic, i.e., same spacing for all dimensions
    :mra_data (boolean): specify the input data

    :return: landmarks in voxel coordinate space and patient coordinate space;
             new spacing and affine matrix for the resampled data under cardiac coordinate space;
             maximum distance for cine-MRI and MRA (where the direction is defined by mvc and apex)
    """

    # 1 lv cavity 
    # 2: lv-myo, 
    # 3:rv, 
    # 4: rv-myo, 
    # 5: la, 
    # 6: ra, 
    # 7: ao,

    # required label values:  LVM-1; LV-2; RV-3; RA-4; LA-5
    seg_original = nib.load(path).get_fdata() 
    #seg = seg_original# np.zeros(seg_original.shape)

    seg = np.zeros(seg_original.shape)
    seg[seg_original == 1] = 2
    seg[seg_original == 2] = 1
    seg[seg_original == 3] = 3
    seg[seg_original == 4] = 5
    seg[seg_original == 5] = 4

    seg_LV_tmp = (seg == 2).astype(seg.dtype)
    # clear unwanted segmentation (find the largest connected component)
    seg_LV = getLargestCC(seg_LV_tmp).astype(seg.dtype)

    seg_RV_tmp = (seg == 3).astype(seg.dtype)
    seg_RV = getLargestCC(seg_RV_tmp).astype(seg.dtype)

    seg_RA_tmp = (seg == 4).astype(seg.dtype)
    seg_RA = getLargestCC(seg_RA_tmp).astype(seg.dtype)

    seg_LA_tmp = (seg == 5).astype(seg.dtype)
    seg_LA = getLargestCC(seg_LA_tmp).astype(seg.dtype)

    seg_binary_tmp = (seg != 0).astype(seg.dtype)
    seg_binary = getLargestCC(seg_binary_tmp).astype(seg.dtype)

    # mitral valve center
    seg_LA_dilation = scipy.ndimage.binary_dilation(seg_LA).astype(seg.dtype)
    mvc = np.mean(np.where(np.multiply(seg_LV, seg_LA_dilation) == 1), axis=1)

    # apex
    LV_points = np.where(seg_LV == 1)
    max_d = float('-inf')
    apex = None
    for i in range(len(LV_points[0])):
        if len(seg.shape) == 3:
            tmp_point = np.array([LV_points[0][i], LV_points[1][i], LV_points[2][i]])
        else:
            tmp_point = np.array([LV_points[0][i], LV_points[1][i]])

        tmp_distance = distance.euclidean(mvc, tmp_point)
        if tmp_distance > max_d :
            max_d = tmp_distance
            apex = tmp_point

    # centroid's of LV and RV
    lvc = np.mean(np.where(seg_LV == 1), axis=1)
    rvc = np.mean(np.where(seg_RV == 1), axis=1)

    # tricuspid valve center
    seg_RA_dilation = scipy.ndimage.binary_dilation(seg_RA).astype(seg.dtype)
    tvc = np.mean(np.where(np.multiply(seg_RV, seg_RA_dilation) == 1), axis=1)

    # center of the heart
    coh = np.mean(np.where(seg_binary == 1), axis=1)

    # transform landmarks into 3D patient coordinate space
    landmarks_list_voxel_space = [apex, mvc, tvc, lvc, rvc, coh]
    landmarks = np.array(landmarks_list_voxel_space).T
    landmarks_mtx = np.ones((4, landmarks.shape[1]))
    landmarks_mtx[:3, :] = landmarks
    affine_mtx = nib.load(path).get_qform()
    landmarks_patient_space = np.dot(affine_mtx, landmarks_mtx)[:3, :]

    apex_ps = landmarks_patient_space[:, 0]
    mvc_ps = landmarks_patient_space[:, 1]
    tvc_ps = landmarks_patient_space[:, 2]
    coh_ps = landmarks_patient_space[:, 5]

    # find the maximum distance of foreground labels (where the direction is defined by mvc and apex)
    # it will be used for defining the new spacing, i.e.,  max_distance* 1.2 / output_size, where 30% is for the background channel

    vec_apex_mvc = apex - mvc
    surface_foreground = scipy.ndimage.binary_dilation(seg_binary).astype(seg.dtype) - seg_binary
    surface_foreground_points = np.where(surface_foreground == 1)
    foreground_points_ls = []
    for i in range(len(surface_foreground_points[0])):
        tmp_foreground_point = np.array(
            [surface_foreground_points[0][i], surface_foreground_points[1][i], surface_foreground_points[2][i]])
        foreground_points_ls.append(tmp_foreground_point)
    foreground_points_array = np.array(foreground_points_ls)
    vec_foreground = foreground_points_array - mvc
    dot_res = np.dot(vec_foreground, vec_apex_mvc)
    #max_idx = np.argmax(dot_res)
    min_idx = np.argmin(dot_res)
    farthest_point = foreground_points_array[min_idx, :]
    # map the point into patient coordinate space
    farthest_point_ps = np.dot(affine_mtx, np.append(farthest_point, 1))[:3]
    max_distance = distance.euclidean(farthest_point_ps, apex_ps)

    vec_three = mvc_ps - apex_ps
    vec_tmp = tvc_ps - mvc_ps
    vec_two_tmp = np.cross(vec_three, vec_tmp)
    vec_two_final = vec_two_tmp/np.linalg.norm(vec_two_tmp)
    vec_one_tmp = np.cross(vec_two_tmp, vec_three)
    vec_one_final = vec_one_tmp/np.linalg.norm(vec_one_tmp)
    vec_three_tmp = np.cross(vec_one_tmp, vec_two_tmp) # recalculate the vector three (more accurate)
    vec_three_final = vec_three_tmp/np.linalg.norm(vec_three_tmp)

    rotation_mtx = np.zeros((3, 3))
    new_affine_mtx = np.eye(4)
    # new spacing for cine-MRI & MRA in cardiac coordinate space, 30% for background
    new_space = (np.round(max_distance) * 1.3) / output_size
    rotation_mtx[:, 0] = vec_one_final * new_space
    rotation_mtx[:, 1] = vec_two_final * new_space
    rotation_mtx[:, 2] = vec_three_final * new_space
    # specify translation
    center_of_voxel = np.array([80, 80, 80])
    translation_col = coh_ps - np.dot(rotation_mtx, center_of_voxel)
    new_affine_mtx[:3, :3] = rotation_mtx
    new_affine_mtx[:3, 3] = translation_col

    return landmarks_list_voxel_space, landmarks_patient_space, new_space, max_distance, new_affine_mtx


# Precompute 3D index grid (shared across workers)
ijk_index = np.stack(np.meshgrid(np.arange(160), np.arange(160), np.arange(160), indexing='ij'), axis=-1)
ijk_mtx = np.ones((160, 160, 160, 4), dtype=np.float64)
ijk_mtx[..., :3] = ijk_index
ijk_mtx_flat = ijk_mtx.reshape(-1, 4).T  # shape (4, 160^3)

def process_case(d, data_path: Path, output_folder: Path):

    data_motion_path = os.path.join(data_path, d)
    nii_files = os.listdir(data_motion_path)
    ref_data_name = '_'.join(d.split('_')[:-1]) + '_1'
    nii_files_ref = os.listdir(os.path.join(data_path, ref_data_name))

    final_3d_sparse_vol = np.zeros((160, 160, 160), dtype=np.float32)

    mri_4ch_file = next(s for s in nii_files_ref if '4ch' in s.lower())
    mri_4ch_path = os.path.join(data_path, ref_data_name, mri_4ch_file)
    _, _, _, _, new_affine_mtx = landmark_gen(mri_4ch_path, output_size=160)

    for nii_file in nii_files:
        
        if '5ch' in nii_file:
            continue

        nii_path_each = os.path.join(data_motion_path, nii_file)
        dicom_info_list = extract_info(nii_path_each, False)
        data_array = dicom_info_list[0]

        nii_file_lower = nii_file.lower()
        if 'sax' in nii_file_lower:
            data_array[(data_array == 4) | (data_array == 5)] = 0
        elif '2ch_lt' in nii_file_lower:
            data_array[(data_array == 3) | (data_array == 4) | (data_array == 6)] = 0
        elif 'rvot' in nii_file_lower:
            data_array[(data_array == 1) | (data_array == 2) | (data_array == 4) | (data_array == 5) | (data_array == 8)] = 0
        elif '3ch' in nii_file_lower:
            data_array[(data_array == 4) | (data_array == 7)] = 0

        data_affine = dicom_info_list[1]
        data_affine[:, 2] /= np.linalg.norm(data_affine[:, 2])

        thickness = 3
        data_with_thickness = np.broadcast_to(data_array[..., 0].T, (thickness, data_array.shape[1], data_array.shape[0]))
        if not data_with_thickness.flags.writeable:
            data_with_thickness = data_with_thickness.copy()

        affine_sparse = new_affine_mtx
        affine_inter = np.linalg.inv(data_affine) @ affine_sparse
        xyz_index_temp = affine_inter @ ijk_mtx_flat
        xyz_index = xyz_index_temp[:3].T.reshape(160, 160, 160, 3)

        input_tensor = torch.from_numpy(data_with_thickness[None, None].astype(np.float32))
        grid = torch.from_numpy(xyz_index.astype(np.float32)[None])

        norm_factor_0 = (input_tensor.shape[2] - 1) / 2
        norm_factor_1 = (input_tensor.shape[3] - 1) / 2
        norm_factor_2 = (input_tensor.shape[4] - 1) / 2

        grid[..., 0] = (grid[..., 0] - norm_factor_2) / (norm_factor_2 + 0.5)
        grid[..., 1] = (grid[..., 1] - norm_factor_1) / (norm_factor_1 + 0.5)
        grid[..., 2] = (grid[..., 2] - norm_factor_0) / (norm_factor_0 + 0.5)

        tmp_img = torch.nn.functional.grid_sample(
            input_tensor, grid, mode='nearest', padding_mode='zeros', align_corners=False
        )[0, 0].numpy()

        final_3d_sparse_vol = np.maximum(tmp_img, final_3d_sparse_vol)

    seg_original = final_3d_sparse_vol
    seg = np.zeros(seg_original.shape)
    seg[seg_original == 1] = 2
    seg[seg_original == 2] = 1
    seg[seg_original == 3] = 3
    seg[seg_original == 4] = 5
    seg[seg_original == 5] = 4
    final_3d_sparse_vol = seg

    label_nifti = nib.Nifti1Image(final_3d_sparse_vol, affine=new_affine_mtx)
    nib.save(label_nifti, os.path.join(output_folder, d + '.nii.gz'))

def to_volume(data_path: Path, output_folder: Path, num_workers: int = 4):
    data_list = sorted([f for f in os.listdir(data_path) if '.json' not in f])

    for d in data_list:
        process_case(d, data_path, output_folder)
    #with Progress(transient=True) as progress:
        #task = progress.add_task(f"Processing {len(data_list)} frames", total=len(data_list))

        #with ProcessPoolExecutor(max_workers=num_workers) as executor:
            #futures = []
            #for d in data_list:
            #    futures.append(executor.submit(process_case, d, data_path, output_folder))

            #for f in futures:
            #    f.result()  # wait for completion
            #    progress.advance(task)


if __name__ == '__main__':

    # ######################################################################################
    parser = argparse.ArgumentParser(description='This script transforms 2D sparse segmentations into 3D sparse volumes ')
    parser.add_argument('-i', '--input', type=Path,
                        help='Folder containing the niftis after slice shifting', default='./corrected_niftis')
    parser.add_argument('-o', '--output_path', type=Path,
                        help='Path where to save the  3D sparse volumes', default='./volume_niftis')

    args = parser.parse_args()

    assert Path(args.input).exists(), \
        f'Cannot not find {args.input}!'

    args.output_path.mkdir(parents=True, exist_ok=True)

    # set list of cases to process
    case_list = os.listdir(args.input)
    case_dirs = [Path(args.input, Path(case).name) for case in case_list]

    log_level = "DEBUG"
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
    logger.info(f"Found {len(case_list)} cases to process.")

    start_time = time.time()

    for case in case_dirs:
        logger.info(f"Processing {os.path.basename(case)}")
        output_nifti = args.output_path / case.name

        Path(output_nifti).mkdir(parents=True, exist_ok=True)
        to_volume(case, output_nifti)


    logger.info(f"Total cases processed: {len(case_dirs)}")
    logger.info(f"Total time: {time.time() - start_time}")
    logger.success(f'Done. Results are saved in {args.output_path}')
