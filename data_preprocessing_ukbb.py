import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
import time


def ukbb_split(data_path):
    data_list = sorted(os.listdir(data_path))
    output_path = os.path.join(os.path.dirname(data_path), 'UKBB_before_SSA')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i in tqdm(range(len(data_list))):
        # format for each patient folder: .../your local path/UKBB/UKBB_patient_ID/Instance_2/nnUNet_segs
        seg_folder = os.path.join(data_path, data_list[i], 'Instance_2/nnUNet_segs')
        seg_list = os.listdir(seg_folder)

        output_path_each_patient = os.path.join(output_path, data_list[i])
        if not os.path.exists(output_path_each_patient):
            os.makedirs(output_path_each_patient)

        for k in range(len(seg_list)):
            seg_path = os.path.join(seg_folder, seg_list[k])
            seg_arr = nib.load(seg_path).get_fdata()
            seg_aff = nib.load(seg_path).affine

            # the last dimension of the shape is the total number of frames
            for l in range(seg_arr.shape[-1]):
                output_path_each_frame = os.path.join(output_path_each_patient, data_list[i] + '_' + str(l + 1))
                # save the new nifti files based on the frame number
                if not os.path.exists(output_path_each_frame):
                    os.makedirs(output_path_each_frame)

                if 'sa' not in seg_list[k].lower():
                    # recalculate the affine matrix, with unit thickness
                    seg_aff[:3, 2] = seg_aff[:3, 2] / np.linalg.norm(seg_aff[:3, 2])
                    # numerical errors and integer values for 3D slicer visualisation
                    tmp_array = np.round(seg_arr[..., l]).astype(np.int16)
                    # 2CH
                    if '2ch' in seg_list[k].lower():
                        new_tmp_array = tmp_array.copy()
                        # relabel LA from 3 to 4
                        new_tmp_array[tmp_array == 3] = 4
                        new_nifti = nib.Nifti1Image(new_tmp_array, seg_aff)
                        nib.save(new_nifti, os.path.join(output_path_each_frame, seg_list[k].replace('.nii.gz',
                                                                                               '_f' + str(
                                                                                                   l + 1) + '_s' + str(
                                                                                                   1) + '.nii.gz')))
                    # 3CH
                    elif '3ch' in seg_list[k].lower():
                        new_tmp_array = tmp_array.copy()
                        # remove label from 5 to 0, i.e., no aorta label
                        new_tmp_array[tmp_array == 5] = 0
                        new_nifti = nib.Nifti1Image(new_tmp_array, seg_aff)
                        nib.save(new_nifti, os.path.join(output_path_each_frame, seg_list[k].replace('.nii.gz',
                                                                                               '_f' + str(
                                                                                                   l + 1) + '_s' + str(
                                                                                                   1) + '.nii.gz')))
                    # 4CH
                    else:
                        new_nifti = nib.Nifti1Image(tmp_array, seg_aff)
                        nib.save(new_nifti, os.path.join(output_path_each_frame, seg_list[k].replace('.nii.gz',
                                                                                               '_f' + str(
                                                                                                   l + 1) + '_s' + str(
                                                                                                   1) + '.nii.gz')))
                # SAX stack
                else:
                    for s in range(seg_arr[..., l].shape[-1]):
                        tmp_sax = np.expand_dims(seg_arr[..., l][..., s], axis=-1)
                        tmp_aff = seg_aff.copy()
                        # recalculate the affine matrix for each slice
                        tmp_aff[:, 3] = tmp_aff[:, 2] * s + tmp_aff[:, 3]
                        tmp_aff[:3, 2] = tmp_aff[:3, 2] / np.linalg.norm(tmp_aff[:3, 2])
                        new_nifti = nib.Nifti1Image(np.round(tmp_sax).astype(np.int16), tmp_aff)
                        nib.save(new_nifti, os.path.join(output_path_each_frame, seg_list[k].replace('.nii.gz',
                                                                                               '_f' + str(
                                                                                                   l + 1) + '_s' + str(
                                                                                                   s + 1) + '.nii.gz')))

    return




if __name__ == '__main__':

    # ######################################################################################
    parser = argparse.ArgumentParser(description='This script converts UKBB segmentation from multi-frame NIFTI format into '
                                                 'individual NIFTI files, with each file containing a single slice segmentation')
    parser.add_argument('-i', '--input', type=Path,
                        help='Folder containing the UKBB segmentation niftis', default='./UKBB')

    args = parser.parse_args()

    assert Path(args.input).exists(), \
        f'Cannot not find {args.input}!'


    start_time = time.time()
    ukbb_split(args.input)
    print("Total running time:" + str(time.time() - start_time) + " seconds")

