import os
import glob as glob
import random
import argparse
from losses import TverskyLoss
# smp
from torch.utils.data import DataLoader
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.data import Dataset
from monai.transforms import (
    LoadPNGd,
    AsChannelFirstd,
    Lambdad,
    Resized,
    ScaleIntensityd,
    RandAdjustContrastd,
    RandGaussianNoised,
    NormalizeIntensityd,
    RandRotate90d,
    RandFlipd,
    RandRotated,
    RandZoomd,
    RandAffined,
    Rand2DElasticd,
    ToTensord
)
from myutils import create_file

def get_argsparser():
    parser = argparse.ArgumentParser("run a model for Kvasir-SEG")
    parser.add_argument("--data_root_dir", type=str, default="/kvasir-seg", help="uo")
    parser.add_argument("--results_dir", type=str, default="/results_dir",
                        help="go")
    parser.add_argument("--train_batch_size", type=int, default=16, help="training batch size")
    parser.add_argument("--cutmix", type=bool, default=True, help="to do cutmix or not")

    # train options
    parser.add_argument("--gpu_id", type=str, default='0,1,2', help='GPU ID')
    parser.add_argument("--encoder_weights", type=str, default='imagenet', help='GPU ID')
    return parser


def get_imagenet_mean_std():
    matrix = np.ones((256, 256))
    R_mean, G_mean, B_mean = 0.485 * matrix, 0.456 * matrix, 0.406 * matrix  # [0.485, 0.456, 0.406]
    R_std, G_std, B_std = 0.229 * matrix, 0.224 * matrix, 0.225 * matrix  # [0.229, 0.224, 0.225]
    mean = np.array([R_mean, G_mean, B_mean])
    std = np.array([R_std, G_std, B_std])
    return mean, std


def get_training_dataset(train_images, train_segs):
    train_files = [{"image": img, "label": seg} for img, seg in zip(train_images, train_segs)]
    print("train_files", len(train_files))
    mean, std = get_imagenet_mean_std()
    train_transform = monai.transforms.Compose(
        [
            LoadPNGd(keys=["image", "label"]),
            AsChannelFirstd(keys=["image", "label"], channel_dim=-1),
            Lambdad(keys=['label'], func=lambda x: x[0:1] * 0.2125 + x[1:2] * 0.7154 + x[2:3] * 0.0721),
            Resized(["image", "label"], spatial_size=(256, 256)),
            ScaleIntensityd(keys=["image", "label"], minv=0.0, maxv=1.0),
            RandAdjustContrastd(keys=["image"], prob=0.3),
            RandGaussianNoised(keys=["image"], prob=0.5),
            RandRotate90d(keys=["image", "label"], prob=0.5),
            RandFlipd(keys=["image", "label"], prob=0.5),
            RandRotated(keys=["image", "label"], range_x=180, range_y=180, prob=0.5),
            RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=1, max_zoom=2),
            RandAffined(keys=["image", "label"], prob=0.5),
            Rand2DElasticd(keys=["image", "label"], magnitude_range=(0, 1), spacing=(0.3, 0.3), prob=0.5),
            NormalizeIntensityd(keys=["image"], subtrahend=mean, divisor=std),
            ToTensord(keys=["image", "label"]),
            Lambdad(keys=['label'], func=lambda x: (x > 0.5).float())
        ]
    )
    return Dataset(train_files, transform=train_transform)


def get_val_dataset(val_images, val_segs):
    val_files = [{"image": img, "label": seg} for img, seg in zip(val_images, val_segs)]
    print("val_files", len(val_files))
    mean, std = get_imagenet_mean_std()
    val_transform = monai.transforms.Compose(
        [
            LoadPNGd(keys=["image", "label"]),
            AsChannelFirstd(keys=["image", "label"], channel_dim=-1),
            Lambdad(keys=['label'], func=lambda x: x[0:1] * 0.2125 + x[1:2] * 0.7154 + x[2:3] * 0.0721),
            Resized(["image", "label"], spatial_size=(256, 256)),
            ScaleIntensityd(keys=["image", "label"], minv=0.0, maxv=1.0),
            NormalizeIntensityd(keys=["image"], subtrahend=mean, divisor=std),
            ToTensord(keys=["image", "label"]),
            Lambdad(keys=['label'], func=lambda x: (x > 0.5).float())
        ]
    )
    return Dataset(val_files, transform=val_transform)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_train_sublist(sublist, train_images, train_segs):
    train_image_list = list()
    train_mask_list = list()
    for i in sublist:
        train_image_list.append(train_images[i])
        train_mask_list.append(train_segs[i])
    return train_image_list, train_mask_list


def train(model, criterion, optimizer, writer, train_loader, device, val_loader, dice_metric, iou_metric, precision_metric, recall_metric, file, cutmix):
    max_score = 0
    for i in range(0, 1000):
        train_epoch_loss = 0
        train_step = 0

        for batch_data in train_loader:
            optimizer.zero_grad()
            train_step += 1
            image, mask = batch_data["image"].to(device), batch_data["label"].to(device)
            if cutmix:
                # generate mixed sample
                lam = np.random.beta(1, 1)
                rand_index = torch.randperm(image.size()[0]).cuda()
                bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
                image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]
                mask[:, :, bbx1:bbx2, bby1:bby2] = mask[rand_index, :, bbx1:bbx2, bby1:bby2]

            output = model(image)
            loss = criterion(mask, output)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
        train_epoch_loss /= train_step

        if i % 5 == 0:
            with torch.no_grad():
                val_step = 0
                val_epoch_loss = 0
                dice_sum = 0
                iou_sum = 0
                precision_sum = 0
                recall_sum = 0

                for batch_data in val_loader:
                    val_step += 1
                    val_image, val_mask = (
                        batch_data["image"].to(device),
                        batch_data["label"].to(device))

                    val_output = model(val_image)
                    loss = criterion(val_mask, val_output)
                    val_epoch_loss += loss.item()
                    val_output = (val_output > 0.5).float()
                    dice = dice_metric(val_output, val_mask)
                    iou = iou_metric(val_output, val_mask)
                    precision = precision_metric(val_output, val_mask)
                    recall = recall_metric(val_output, val_mask)
                    dice_sum += dice
                    iou_sum += iou
                    precision_sum += precision
                    recall_sum += recall

                val_dice_mean = dice_sum / val_step
                val_iou_mean = iou_sum / val_step
                val_epoch_loss /= val_step
                precision_mean = precision_sum/val_step
                recall_mean = recall_sum/val_step

                writer.add_scalar("train_epoch_loss", train_epoch_loss, i)
                writer.add_scalar("val_epoch_loss", val_epoch_loss, i)
                writer.add_scalar("val_dice_metric", val_dice_mean, i)
                writer.add_scalar("val_precision", precision_mean, i)
                writer.add_scalar("val_recall", recall_mean, i)

                if max_score < val_dice_mean:
                    max_score = val_dice_mean
                    checkpoint = {"state_dict": model,
                                  "epoch": i,
                                  "fscore": val_dice_mean,
                                  "iou": val_iou_mean,
                                  "optimizer": optimizer.state_dict()
                                  }
                    torch.save(checkpoint, file)
                    print('Model saved!')


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = args.data_root_dir
    results_dir = args.results_dir

    train_images = sorted(glob.glob(os.path.join(data_path, "trainset", "images", "*.jpg")))
    train_segs = sorted(glob.glob(os.path.join(data_path, "trainset", "masks", "*.jpg")))
    val_images = sorted(glob.glob(os.path.join(data_path, "valset", "images", "*.jpg")))
    val_segs = sorted(glob.glob(os.path.join(data_path, "valset", "masks", "*.jpg")))

    image_indexes = list(range(0, len(train_images)))
    random.shuffle(image_indexes)
    x, y = [image_indexes[i::2] for i in range(2)]

    train_image_list1, train_mask_list1  = get_train_sublist(x, train_images, train_segs)
    train_image_list2, train_mask_list2 = get_train_sublist(y, train_images, train_segs)

    train_dataset1 = get_training_dataset(train_image_list1, train_mask_list1)
    train_dataset2 = get_training_dataset(train_image_list2, train_mask_list2)
    valid_dataset = get_val_dataset(val_images, val_segs)

    ENCODER1 = "ResNet-34"
    ENCODER2 = "EfficientNet-B2"

    ENCODER_WEIGHTS = "imagenet"
    CLASSES = ['polyp']
    ACTIVATION = 'sigmoid'

    # model
    model1 = smp.Unet(encoder_name=ENCODER1, encoder_weights=ENCODER_WEIGHTS, classes=len(CLASSES),
                      activation=ACTIVATION).to(device)
    model2 = smp.Unet(encoder_name=ENCODER2, encoder_weights=ENCODER_WEIGHTS, classes=len(CLASSES),
                      activation=ACTIVATION).to(device)

    train_loader1 = DataLoader(train_dataset1, batch_size=args.train_batch_size, shuffle=True, num_workers=12)
    train_loader2 = DataLoader(train_dataset2, batch_size=args.train_batch_size, shuffle=True, num_workers=12)
    val_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    #metrics
    dice_metric = smp.utils.metrics.Fscore(activation="sigmoid", threshold=0.5)
    iou_metric = smp.utils.metrics.IoU(activation="sigmoid", threshold=0.5)
    precision_metric = smp.utils.metrics.Precision(activation="sigmoid", threshold=0.5)
    recall_metric = smp.utils.metrics.Recall(activation="sigmoid", threshold=0.5)
    model_list = [model1, model2]
    train_loader_list = [train_loader1, train_loader2]
    encoder_list = [ENCODER1, ENCODER2]

    for index in range(len(model_list)):
        model = model_list[index]
        train_loader = train_loader_list[index]
        encoder = encoder_list[index]
        a = np.linspace(0.9, 1, 100)
        criterion = TverskyLoss(beta=np.random.choice(a))
        file = os.path.join(results_dir, encoder + 'best_model.pth')
        create_file(file)
        summary_writer_dir = os.path.join(results_dir, "runs/" + encoder + "/train-val")
        writer = SummaryWriter(summary_writer_dir)
        optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])
        cutmix = args.cutmix

        train(model, criterion, optimizer, encoder, writer, train_loader, device, val_loader, dice_metric, iou_metric, precision_metric, recall_metric, file, cutmix)


if __name__ == '__main__':
    args = get_argsparser().parse_args()
    main(args)
