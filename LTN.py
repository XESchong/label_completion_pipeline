import os
import numpy as np
import random
import pandas as pd
#from skimage.measure import centroid
import nibabel as nib
from tqdm import tqdm
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import SimpleITK as sitk
from rich.progress import Progress
from pathlib import Path
from scipy.ndimage import label
import gc

# network
class UNet3d(nn.Module):
    def contracting_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(out_channels),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ConvTranspose3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1)
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            #torch.nn.Sigmoid()
        )
        return block

    def __init__(self, in_channel, out_channel):
        super(UNet3d, self).__init__()
        # Encode
        self.conv_encode1 = self.contracting_block(in_channel, 16, 32)
        self.conv_maxpool1 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(32, 32, 64)
        self.conv_maxpool2 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(64, 64, 128)
        self.conv_maxpool3 = torch.nn.MaxPool3d(kernel_size=2)
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=3, in_channels=128, out_channels=128, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(128),
            torch.nn.Conv3d(kernel_size=3, in_channels=128, out_channels=256, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm3d(256),
            torch.nn.ConvTranspose3d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1,
                                     output_padding=1)
        )
        # Decode
        self.conv_decode3 = self.expansive_block(128+256, 128, 128)
        self.conv_decode2 = self.expansive_block(64+128, 64, 64)
        self.final_layer = self.final_block(32+64, 32, out_channel)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=False)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=False)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=False)
        final_layer = self.final_layer(decode_block1)
        return final_layer


def getLargestCC(segmentation):
    labels, num_features = label(segmentation)
    
    if num_features == 0:
        # No connected components found
        return np.zeros_like(segmentation, dtype=bool)

    # Compute sizes of connected components
    component_sizes = np.bincount(labels.flat)[1:]  # skip background count (label 0)
    largest_component = np.argmax(component_sizes) + 1  # +1 because label 0 is background
    largestCC = labels == largest_component
    return largestCC

@torch.no_grad()
def eval_model(model_file: Path, cvs_path: Path, output_LTN: Path, out_channel: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    unet = UNet3d(in_channel=6, out_channel=out_channel)
    checkpoint = torch.load(model_file, map_location=device)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.to(device)

    # Read input file list
    test_x_csv = pd.read_csv(cvs_path / '3d_sparse_oh.csv')
    image_paths = [Path(p) for p in test_x_csv['img'].tolist()]

    for i in range(0, len(image_paths)):
        batch_paths = image_paths[i:i+1]
        batch_images = []
        original_sitk = []

        # Load and preprocess
        for path in batch_paths:
            img = sitk.ReadImage(str(path))
            original_sitk.append(img)
            arr = sitk.GetArrayFromImage(img)[np.newaxis, ...]  # shape (1, D, H, W, C)
            batch_images.append(arr)

        batch_array = np.concatenate(batch_images, axis=0)  # shape (B, D, H, W, C)
        batch_tensor = torch.from_numpy(batch_array).permute(0, 4, 1, 2, 3).float().to(device)  # B, C, D, H, W

        # Inference
        with torch.amp.autocast(device_type="cuda", enabled=False):  # force full precision
            outputs = unet(batch_tensor).cpu().numpy()  # [B, C, D, H, W]

        # Post-process
        for j, output in enumerate(outputs):
            label_pred = np.argmax(output, axis=0)  # [D, H, W]

            new_label = np.zeros_like(label_pred, dtype=np.uint8)
            for k in range(out_channel):
                seg = (label_pred == k).astype(np.uint8)
                seg_largest = getLargestCC(seg)
                new_label[seg_largest == 1] = k

            result_img = sitk.GetImageFromArray(new_label)
            result_img.CopyInformation(original_sitk[j])

            out_name = batch_paths[j].name
            sitk.WriteImage(result_img, output_LTN / out_name)

    return