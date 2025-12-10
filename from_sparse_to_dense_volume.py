import os
import time
import numpy as np
import SimpleITK as sitk
import pandas as pd
from scipy import ndimage
from pathlib import Path
from loguru import logger
import argparse
import shutil

#import concurrent.futures
#from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

from LTN import eval_model

def gaussian_filter_gpu(mask, sigma):
    arr_gpu = cp.asarray(mask)
    filtered_gpu = cp.ndimage.gaussian_filter(arr_gpu, sigma=sigma)
    return cp.asnumpy(filtered_gpu)

def one_hot_labelmap_with_mask_fast(path, smoothing_sigma=0, save_path=None, use_gpu=False):
    labelmap = sitk.ReadImage(path, sitk.sitkInt64)
    lab_array = sitk.GetArrayFromImage(labelmap)
    labels = np.unique(lab_array)
    labels.sort()

    h, w, d = lab_array.shape
    lab_array_one_hot = np.zeros((h, w, d, labels.size), dtype=np.float32)

    for idx, lab in enumerate(labels):
        mask = (lab_array == lab).astype(np.float32)
        if smoothing_sigma > 0:
            mask = gaussian_filter_gpu(mask, sigma=smoothing_sigma) if use_gpu else ndimage.gaussian_filter(mask, sigma=smoothing_sigma, mode='nearest')
        lab_array_one_hot[..., idx] = mask

    labelmap_one_hot = sitk.GetImageFromArray(lab_array_one_hot, isVector=True)
    labelmap_one_hot.CopyInformation(labelmap)

    output_file = save_path / os.path.basename(path)
    sitk.WriteImage(labelmap_one_hot, str(output_file))
    return str(output_file)

def process_label_file_worker(path, output_path, smoothing_sigma=0, use_gpu=False):
    try:
        return one_hot_labelmap_with_mask_fast(path, smoothing_sigma, save_path=output_path, use_gpu=use_gpu)
    except Exception as e:
        logger.warning(f"Failed to process {path}: {e}")
        return None

def to_one_hot(label_folder, output_path, is_sparse=True, max_workers=8, use_gpu=False):
    volume_type = 'sparse' if is_sparse else 'dense'
    file_paths = sorted([str(Path(label_folder) / f) for f in os.listdir(label_folder) if f.endswith('.nii.gz')])

    output_files = []
    for f in file_paths:
        result = process_label_file_worker(f, output_path, 0, use_gpu)
        output_files.append(result)

    #with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #    futures = {
    #        executor.submit(process_label_file_worker, f, output_path, 0, use_gpu): f
    #        for f in file_paths
    #    }
    #    output_files = []
    #    for future in as_completed(futures):
    #        result = future.result()
    #        if result:
    #            output_files.append(result)

    # Save processed list
    pd.DataFrame({'img': output_files}).to_csv(output_path / f'3d_{volume_type}_oh.csv', index=False)

def process_case(case: Path, output_path: Path, path_ltn: Path, max_workers: int, use_gpu: bool):
    #logger.info(f"Processing case: {case.name}")
    output_case_dir = output_path / case.name
    output_case_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: convert to one-hot
    to_one_hot(case, output_case_dir, max_workers=max_workers, use_gpu=use_gpu)

    # Step 2: apply LTN
    logger.info(f"Running LTN for {case.name}")
    output_LTN = output_path / f"{case.name}_LTN"
    output_LTN.mkdir(parents=True, exist_ok=True)

    eval_model(path_ltn, output_case_dir, output_LTN, out_channel=11)
    shutil.rmtree(output_case_dir)

def main():
    parser = argparse.ArgumentParser(description='Convert 2D sparse volumes to 3D one-hot encoded volumes')
    parser.add_argument('-i', '--input', type=Path, default='./volume_niftis', help='Folder containing sparse volumes')
    parser.add_argument('-o', '--output_path', type=Path, default='./volume_niftis_oh', help='Folder to save output')
    parser.add_argument('-ltn_pth', '--path_ltn', type=Path, default='./models/LabelTransferNetwork/LTN_params.pth', help='Path to LTN checkpoint')
    parser.add_argument('--max_workers', type=int, default=1, help='Max number of worker threads')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for filtering (requires CuPy)')
    args = parser.parse_args()

    if args.use_gpu and not GPU_AVAILABLE:
        logger.warning("GPU requested but CuPy is not installed. Falling back to CPU.")
        args.use_gpu = False

    assert args.input.exists(), f'Input path not found: {args.input}'
    assert args.path_ltn.exists(), f'LTN checkpoint not found: {args.path_ltn}'
    args.output_path.mkdir(parents=True, exist_ok=True)

    case_dirs = [Path(args.input, c) for c in sorted(os.listdir(args.input)) if Path(args.input, c).is_dir()]
    logger.info(f"Found {len(case_dirs)} cases to process.")
    start_time = time.time()

    # Process cases in parallel
    #with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    #    futures = [
    #        executor.submit(process_case, case, args.output_path, args.path_ltn, args.max_workers, args.use_gpu)
    #        for case in case_dirs
    #    ]
    #    for f in concurrent.futures.as_completed(futures):
    #        try:
    #            f.result()
    #        except Exception as e:
    #            logger.error(f"Error during processing: {e}")
    for case in case_dirs:
        logger.info(f"Processing case: {case.name}")
        #output_case_dir = args.output_path / case.name
        #output_case_dir.mkdir(parents=True, exist_ok=True)

        process_case(case, args.output_path, args.path_ltn, args.max_workers, args.use_gpu)

    elapsed = time.time() - start_time
    logger.success(f"Done. Processed {len(case_dirs)} cases in {elapsed:.2f} seconds.")

if __name__ == '__main__':
    main()
