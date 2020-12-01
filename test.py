import os
import time
import argparse
import numpy as np
import glob as glob
from skimage import io

# torch
import torch
from torch.utils.data import DataLoader

# monai
import monai  # pip install git+https://github.com/Project-MONAI/MONAI#egg=MONAI
from monai.transforms import (
    LoadPNGd,
    AsChannelFirstd,
    Resized,
    ScaleIntensityd,
    NormalizeIntensityd,
    ToTensord
)


# smp library
import segmentation_models_pytorch as smp  # pip install git+https://github.com/qubvel/segmentation_models.pytorch


def get_argsparser():
    parser = argparse.ArgumentParser("test arguments")
    parser.add_argument("--data_root_dir", type=str, default="/medico2020", help="test data path")
    parser.add_argument("--checkpoint1", type=str, default="ResNet-34_best_model.pth", help="test data path")
    parser.add_argument("--checkpoint2", type=str, default="EfficientNet-B2_best_model.pth", help="test data path")
    parser.add_argument("--output_fpath", type=str, default="text.txt", help="output file path")
    parser.add_argument("--model", type=str, default="resnet34_unet",
                        help="training encoders and decoders")
    parser.add_argument("--test_batch_size", type=int, default=1, help="batch size for test")
    return parser


def write_to_file(path, data, mode='w'):
    file = open(path, mode)
    file.write(data)
    file.close()


def create_file(filepath):
    try:
        print("creating file", filepath)
        open(filepath, 'w').close()
    except OSError:
        print(f"Error: creating file with name {filepath}")


def get_imagenet_mean_std():
    matrix = np.ones((256, 256))
    R_mean, G_mean, B_mean = 0.485 * matrix, 0.456 * matrix, 0.406 * matrix  # [0.485, 0.456, 0.406]
    R_std, G_std, B_std = 0.229 * matrix, 0.224 * matrix, 0.225 * matrix  # [0.229, 0.224, 0.225]
    mean = np.array([R_mean, G_mean, B_mean])
    std = np.array([R_std, G_std, B_std])
    return mean, std


def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_images_path = args.data_root_dir
    output_fpath = args.output_fpath

    # get images list
    test_images = sorted(glob.glob(os.path.join(test_images_path, "*.jpg")))

    # for normalizing
    mean, std = get_imagenet_mean_std()

    test_transform = monai.transforms.Compose(
        [
            LoadPNGd(keys=["image"]),
            AsChannelFirstd(keys=["image"], channel_dim=-1),
            Resized(["image"], spatial_size=(256, 256)),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            NormalizeIntensityd(keys=["image"], subtrahend=mean, divisor=std),
            ToTensord(keys=["image"])
        ]
    )

    test_files = [{"image": img} for img in zip(test_images)]

    # transforms
    test_dataset = monai.data.Dataset(test_files, transform=test_transform)

    # get dataloader
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # load model

    # load model
    ckpt1 = torch.load(args.checkpoint1)
    ckpt2 = torch.load(args.checkpoint2)

    model1 = ckpt1["state_dict"]
    model2 = ckpt2["state_dict"]
    model_name = "mix"

    # create output file
    create_file(output_fpath)
    precision = 0
    recall = 0
    f2_score = 0
    jaccard = 0
    dice = 0
    accuracy = 0
    FPS = 0

    with torch.no_grad():
        start = int(round(time.time()))
        for batch_data in test_loader:
            test_data = batch_data["image"].to(device)
            test_output1 = model1(test_data)
            test_output2 = model2(test_data)
            test_output1 = (test_output1>0.5).float()
            test_output2 = (test_output2>0.5).float()
            test_output = test_output1*test_output2

        end = int(round(time.time()))
        mean_time = (end - start) / len(test_images)
        print(mean_time)
        write_to_file(output_fpath, str(mean_time))


if __name__ == "__main__":
    args = get_argsparser().parse_args()
    test(args)
