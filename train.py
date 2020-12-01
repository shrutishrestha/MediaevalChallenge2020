# preinstallation of monai required
from torchvision.utils import save_image
from collections import defaultdict
import numpy as np
import shutil
import csv
from tqdm import tqdm as tqdm
import os
import gc
import glob as glob
import cv2
import random
from monai.metrics import DiceMetric
import argparse
import torch
import torch.nn as nn
from losses import TverskyLoss
# smp
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from monai.metrics import DiceMetric
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torchvision.transforms import Normalize, ToTensor, Resize, ToPILImage
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.losses import DiceLoss
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


def get_imagenet_mean_std():
    matrix = np.ones((256, 256))
    R_mean, G_mean, B_mean = 0.485 * matrix, 0.456 * matrix, 0.406 * matrix  # [0.485, 0.456, 0.406]
    R_std, G_std, B_std = 0.229 * matrix, 0.224 * matrix, 0.225 * matrix  # [0.229, 0.224, 0.225]
    mean = np.array([R_mean, G_mean, B_mean])
    std = np.array([R_std, G_std, B_std])
    return mean, std


def get_argsparser():
    parser = argparse.ArgumentParser("run a model for Kvasir-SEG")
    parser.add_argument("--data_root_dir", type=str, default="/home/santosh/PycharmProjects/try/new_data", help="uo")
    parser.add_argument("--results_dir", type=str, default="/home/santosh/PycharmProjects/try/results_dir/ru_e0u_e2p",
                        help="go")
    parser.add_argument("--train_batch_size", type=int, default=16, help="training batch size")
    # train options
    parser.add_argument("--gpu_id", type=str, default='0,1,2', help='GPU ID')
    parser.add_argument("--encoder_weights", type=str, default='imagenet', help='GPU ID')

    return parser


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

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = args.data_root_dir

    train_images = sorted(glob.glob(os.path.join(data_path, "totalsett", "images", "*.jpg")))
    train_segs = sorted(glob.glob(os.path.join(data_path, "totalsett", "masks", "*.jpg")))
    val_images = sorted(glob.glob(os.path.join(data_path, "valset", "images", "*.jpg")))
    val_segs = sorted(glob.glob(os.path.join(data_path, "valset", "masks", "*.jpg")))
    print(len(val_images))
    image_index = list(range(0, len(train_images)))
    random.shuffle(image_index)
    #x, z = [image_index[i::2] for i in range(2)]
    x=0
    y=0
    z=[424, 536, 630, 381, 224, 80, 0, 49, 407, 469, 807, 610, 593, 716, 785, 606, 642, 115, 68, 214, 713, 309, 779, 442, 412, 297, 395, 865, 29, 444, 109, 722, 217, 143, 33, 728, 697, 431, 122, 87, 810, 631, 801, 159, 570, 282, 805, 949, 74, 353, 176, 942, 787, 101, 977, 391, 293, 383, 380, 18, 527, 203, 303, 265, 754, 620, 918, 345, 156, 720, 781, 672, 845, 864, 894, 213, 719, 177, 192, 155, 495, 149, 674, 335, 921, 274, 477, 298, 974, 612, 194, 851, 448, 316, 41, 784, 403, 692, 196, 290, 184, 350, 677, 169, 312, 828, 733, 59, 466, 178, 562, 429, 65, 419, 473, 387, 734, 233, 63, 830, 957, 257, 58, 26, 46, 369, 667, 256, 547, 744, 682, 770, 349, 447, 572, 78, 294, 262, 388, 750, 776, 556, 452, 966, 576, 474, 455, 896, 628, 575, 626, 392, 506, 227, 665, 195, 75, 464, 458, 525, 780, 244, 98, 824, 820, 301, 315, 379, 704, 812, 95, 906, 723, 533, 399, 687, 802, 735, 763, 365, 360, 623, 119, 924, 625, 806, 56, 658, 662, 255, 532, 857, 16, 816, 484, 926, 827, 815, 487, 163, 829, 782, 804, 850, 627, 890, 324, 157, 831, 433, 374, 842, 546, 411, 537, 397, 751, 270, 352, 104, 340, 243, 638, 848, 275, 66, 414, 909, 666, 235, 613, 327, 258, 597, 690, 975, 222, 481, 453, 142, 883, 583, 61, 354, 960, 793, 513, 927, 929, 762, 941, 876, 511, 496, 408, 358, 218, 946, 92, 790, 602, 499, 586, 471, 445, 753, 497, 400, 470, 432, 585, 346, 855, 796, 836, 879, 783, 54, 269, 295, 950, 881, 915, 8, 611, 773, 242, 116, 852, 305, 846, 668, 492, 912, 803, 752, 394, 62, 911, 840, 260, 341, 308, 875, 743, 171, 791, 493, 83, 437, 461, 211, 903, 128, 283, 105, 517, 71, 28, 385, 961, 292, 560, 684, 969, 596, 676, 127, 337, 512, 948, 757, 814, 798, 376, 82, 653, 215, 656, 633, 103, 172, 251, 357, 229, 749, 730, 296, 13, 534, 673, 862, 819, 522, 377, 64, 338, 99, 404, 271, 688, 714, 291, 491, 279, 939, 402, 838, 808, 416, 479, 755, 797, 740, 689, 90, 590, 972, 867, 870, 856, 451, 595, 725, 40, 652, 705, 67, 468, 643, 913, 363, 841, 538, 285, 434, 182, 853, 314, 361, 944, 302, 693, 405, 364, 761, 252, 727, 370, 23, 737, 96, 147, 209, 558, 287, 789, 548, 228, 893, 519, 406, 216, 450, 304, 326, 559, 756, 140, 409, 230, 456, 88, 887, 220, 545, 616, 514, 759, 204, 885, 427, 873, 443, 923, 767, 600, 964, 589, 446, 48, 134, 892, 771, 266, 739, 12, 678, 973, 717, 398, 561, 53, 614, 488, 145, 421, 246, 564, 35, 187, 772, 557, 637, 45, 549, 661, 89, 515, 52, 423, 34, 212, 905, 237, 578, 299, 849, 362, 436]

    #a = [684, 798, 771, 810, 758, 385, 747, 630, 425, 685, 246, 337, 93, 112, 357, 413, 113, 45, 237, 727, 146, 155, 853, 264, 142, 440, 578, 510, 288, 330, 715, 364, 759, 321, 434, 722, 329, 430, 92, 423, 492, 874, 579, 623, 350, 487, 411, 617, 60, 450, 8, 7, 226, 224, 77, 471, 388, 620, 287, 304, 679, 27, 678, 14, 564, 311, 687, 516, 144, 753, 137, 335, 228, 783, 258, 766, 22, 214, 604, 436, 716, 403, 96, 446, 44, 361, 615, 62, 709, 696, 455, 63, 851, 830, 365, 376, 832, 98, 300, 313, 597, 447, 841, 29, 735, 818, 240, 395, 532, 266, 631, 636, 406, 724, 197, 175, 89, 336, 374, 278, 823, 681, 826, 875, 427, 217, 271, 541, 108, 544, 76, 559, 761, 876, 253, 713, 248, 781, 78, 629, 71, 511, 764, 479, 466, 268, 698, 306, 467, 596, 293, 822, 419, 12, 303, 417, 568, 332, 670, 150, 148, 493, 624, 612, 514, 872, 719, 69, 825, 173, 422, 551, 583, 134, 176, 772, 277, 97, 399, 858, 384, 250, 206, 749, 26, 570, 68, 102, 116, 794, 145, 869, 800, 650, 445, 153, 558, 538, 745, 85, 170, 328, 74, 594, 367, 718, 813, 216, 230, 833, 238, 827, 462, 326, 225, 127, 82, 458, 107, 555, 233, 576, 435, 418, 589, 608, 780, 432, 864, 91, 156, 48, 689, 282, 259, 506, 353, 163, 125, 299, 53, 733, 461, 183, 443, 671, 815, 414, 628, 606, 99, 705, 280, 762, 405, 87, 859, 676, 54, 497, 501, 70, 51, 3, 409, 755, 55, 729, 242, 824, 495, 239, 52, 36, 861, 286, 292, 120, 121, 845, 331, 542, 707, 33, 524, 132, 686, 310, 586, 743, 658, 611, 431, 571, 607, 235, 465, 381, 848, 192, 194, 633, 309, 46, 855, 785, 763, 80, 66, 50, 263, 138, 867, 695, 178, 873, 599, 283, 784, 118, 305, 17, 667, 166, 738, 324, 270, 131, 699, 474, 453, 341, 756, 35, 382, 105, 232, 645, 47, 117, 449, 209, 210, 21, 377, 632, 358, 188, 28, 683, 527, 790, 819, 550, 646, 502, 779, 366, 754, 694, 717, 275, 244, 731, 229, 475, 659, 212, 517, 407, 503, 590, 809, 109, 565, 457, 799, 562, 203, 356, 389, 557, 429, 167, 160, 525, 625, 543, 154, 635, 553, 647, 373, 420, 468, 742, 75, 37, 39, 362, 316, 202, 878, 804, 587, 84, 346, 600, 187, 298, 828, 25, 746, 626, 391, 208, 540, 343, 652, 706, 101, 354, 807, 677, 276, 260, 371, 603, 351, 190, 114, 126, 601, 526, 744, 380, 219, 831, 820, 32]

    #b = [792, 115, 196, 441, 72, 412, 531, 172, 20, 838, 618, 770, 857, 2, 720, 691, 57, 788, 653, 793, 704, 124, 464, 182, 243, 549, 400, 383, 347, 308, 387, 593, 294, 164, 806, 135, 476, 748, 732, 649, 256, 94, 534, 552, 110, 673, 61, 692, 348, 301, 739, 563, 314, 168, 513, 119, 741, 556, 654, 272, 802, 657, 499, 693, 86, 165, 547, 49, 622, 59, 340, 79, 642, 227, 776, 711, 104, 149, 334, 585, 64, 829, 83, 396, 320, 836, 614, 438, 222, 786, 255, 284, 10, 439, 451, 442, 281, 90, 847, 279, 478, 38, 199, 452, 338, 252, 345, 333, 782, 379, 349, 481, 254, 835, 846, 296, 641, 567, 159, 136, 877, 231, 215, 812, 302, 477, 500, 469, 40, 710, 750, 133, 494, 261, 392, 656, 402, 702, 602, 327, 152, 257, 591, 352, 147, 369, 11, 180, 638, 737, 821, 143, 236, 643, 472, 871, 627, 189, 198, 415, 849, 339, 408, 186, 201, 535, 220, 323, 218, 463, 791, 669, 0, 460, 213, 355, 342, 416, 262, 207, 141, 760, 158, 269, 561, 795, 751, 372, 515, 850, 490, 616, 195, 65, 398, 103, 249, 808, 537, 634, 297, 736, 251, 312, 816, 860, 504, 789, 193, 621, 424, 536, 325, 211, 204, 410, 662, 24, 177, 610, 844, 560, 584, 581, 619, 265, 725, 454, 668, 856, 491, 318, 267, 95, 595, 151, 708, 803, 870, 245, 390, 368, 774, 498, 393, 421, 545, 672, 221, 787, 448, 752, 529, 539, 181, 485, 179, 637, 433, 548, 247, 484, 840, 317, 100, 157, 205, 577, 285, 23, 879, 5, 130, 797, 174, 664, 184, 865, 378, 868, 598, 674, 554, 866, 648, 9, 507, 842, 528, 273, 768, 777, 43, 801, 854, 572, 394, 666, 640, 512, 796, 682, 1, 661, 274, 428, 660, 522, 31, 34, 837, 613, 88, 566, 862, 459, 588, 344, 67, 700, 805, 363, 734, 580, 42, 488, 811, 489, 675, 523, 740, 728, 765, 15, 322, 359, 701, 13, 30, 16, 480, 644, 122, 397, 769, 58, 509, 4, 291, 651, 171, 19, 680, 690, 723, 191, 697, 730, 817, 574, 111, 663, 834, 185, 843, 483, 520, 767, 370, 386, 655, 505, 140, 726, 703, 863, 546, 665, 6, 444, 18, 605, 200, 289, 375, 508, 169, 482, 582, 106, 852, 518, 569, 437, 575, 473, 162, 775, 530, 128, 241, 609, 592, 129, 360, 519, 315, 223, 521, 290, 401, 757, 470, 496, 234, 814, 319, 714, 307, 81, 639, 533, 73, 56, 688, 712, 41, 456, 721, 839, 404, 486, 778, 573, 139, 426, 161, 123, 295, 773]
    print("x",x)
    print("z",z)

    y = x

    train_image_list1 = list()
    train_image_list2 = list()
    train_image_list3 = list()
    train_mask_list1 = list()
    train_mask_list2 = list()
    train_mask_list3 = list()

    for i in x:
        train_image_list1.append(train_images[i])
        train_mask_list1.append(train_segs[i])

    for j in y:
        train_image_list2.append(train_images[j])
        train_mask_list2.append(train_segs[j])

    for k in z:
        train_image_list3.append(train_images[k])
        train_mask_list3.append(train_segs[k])
    print("train_mask_list2",len(train_mask_list2))
    train_dataset1 = get_training_dataset(train_image_list1, train_mask_list1)
    train_dataset2 = get_training_dataset(train_image_list2, train_mask_list2)
    train_dataset3 = get_training_dataset(train_image_list3, train_mask_list3)

    valid_dataset = get_val_dataset(val_images, val_segs)

    ENCODER1 = "resnet34"
    ENCODER2 = "efficientnet-b0"
    ENCODER3 = "efficientnet-b2"

    ENCODER_WEIGHTS = "imagenet"
    CLASSES = ['polyp']
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multicalss segmentation

    # model
    model1 = smp.Unet(encoder_name=ENCODER1, encoder_weights=ENCODER_WEIGHTS, classes=len(CLASSES),
                      activation=ACTIVATION).to(device)
    model2 = smp.Unet(encoder_name=ENCODER2, encoder_weights=ENCODER_WEIGHTS, classes=len(CLASSES),
                      activation=ACTIVATION).to(device)
    model3 = smp.Unet(encoder_name=ENCODER3, encoder_weights=ENCODER_WEIGHTS, classes=len(CLASSES),
                      activation=ACTIVATION).to(device)
    model4 = smp.PSPNet(encoder_name=ENCODER1, encoder_weights=ENCODER_WEIGHTS, classes=len(CLASSES),
                        activation=ACTIVATION).to(device)
    model5 = smp.PSPNet(encoder_name=ENCODER2, encoder_weights=ENCODER_WEIGHTS, classes=len(CLASSES),
                        activation=ACTIVATION).to(device)
    model6 = smp.PSPNet(encoder_name=ENCODER3, encoder_weights=ENCODER_WEIGHTS, classes=len(CLASSES),
                        activation=ACTIVATION).to(device)

    train_loader1 = DataLoader(train_dataset1, batch_size=args.train_batch_size, shuffle=True, num_workers=12)
    train_loader2 = DataLoader(train_dataset2, batch_size=args.train_batch_size, shuffle=True, num_workers=12)
    train_loader3 = DataLoader(train_dataset3, batch_size=args.train_batch_size, shuffle=True, num_workers=12)
    train_loader4 = DataLoader(train_dataset1, batch_size=args.train_batch_size, shuffle=True, num_workers=12)
    train_loader5 = DataLoader(train_dataset2, batch_size=args.train_batch_size, shuffle=True, num_workers=12)
    train_loader6 = DataLoader(train_dataset3, batch_size=args.train_batch_size, shuffle=True, num_workers=12)

    val_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    #
    # #loss
    a = np.linspace(0.9, 1, 100)
    loss1 = TverskyLoss(beta=np.random.choice(a))
    #loss1 = DiceLoss(sigmoid=True)
    beta = np.random.choice(a)
    loss2 = TverskyLoss(beta=beta)
    loss3 = TverskyLoss(beta=np.random.choice(a))
    loss4 = TverskyLoss(beta=np.random.choice(a))
    loss5 = TverskyLoss(beta=np.random.choice(a))
    loss6 = TverskyLoss(beta=np.random.choice(a))

    metrics = [smp.utils.metrics.Fscore(threshold=0.5), smp.utils.metrics.IoU(threshold=0.5)]
    optimizer1 = torch.optim.Adam([dict(params=model1.parameters(), lr=0.0001)])
    optimizer2 = torch.optim.Adam([dict(params=model2.parameters(), lr=0.0001)])
    optimizer3 = torch.optim.Adam([dict(params=model3.parameters(), lr=0.0001)])
    optimizer4 = torch.optim.Adam([dict(params=model4.parameters(), lr=0.0001)])
    optimizer5 = torch.optim.Adam([dict(params=model5.parameters(), lr=0.0001)])
    optimizer6 = torch.optim.Adam([dict(params=model6.parameters(), lr=0.0001)])

    dice_metric = smp.utils.metrics.Fscore(activation="sigmoid", threshold=0.5)
    iou_metric = smp.utils.metrics.IoU(activation="sigmoid", threshold=0.5)
    precision_metric = smp.utils.metrics.Precision(activation="sigmoid", threshold=0.5)
    recall_metric = smp.utils.metrics.Recall(activation="sigmoid", threshold=0.5)
    models = ["resnet34_unet", "effb0_unet", "effb2_unet", "resnet34_pspnet", "effb0_pspnet", "effb2_pspnet"]

    model_name = "resnet34_unet"
    max_score = 0
    train_loader = train_loader1
    model = model1

    criterion = loss1
    optimizer = optimizer1
    file = os.path.join(args.results_dir, model_name + '_a_beta_'+str(beta)+'_totalset_tversky_cutmix_best_model.pth')
    summary_writer_dir = os.path.join(args.results_dir, "runs/" + model_name + "_totalset_a_beta_"+str(beta)+"_tversky_cutmix/train-val")
    writer = SummaryWriter(summary_writer_dir)

    if not os.path.exists(file):
        print("creating file", file)
        open(file, 'w').close()
    print("file name", file)
    #else:
        #new_split = file.split(".")
        #fileprefix= new_split[0]
        #file = fileprefix+"_new.pth"
        #print("creating file", file)
        #open(file, 'w').close()


    for i in range(0, 1000):
        train_epoch_loss = 0
        train_step = 0

        for batch_data in train_loader:
            optimizer.zero_grad()
            train_step += 1
            image, mask = batch_data["image"].to(device), batch_data["label"].to(device)

            # generate mixed sample
            lam = np.random.beta(1, 1)
            rand_index = torch.randperm(image.size()[0]).cuda()  # batch size [4,1,0,2,5,3,7,6]
            bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
            image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]
            mask[:, :, bbx1:bbx2, bby1:bby2] = mask[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio

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
                dice_monai_sum = 0

                for batch_data in val_loader:
                    val_step += 1
                    val_image, val_mask = (
                        batch_data["image"].to(device),
                        batch_data["label"].to(device))

                    val_output = model(val_image)

                    loss = criterion( val_mask, val_output)

                    val_epoch_loss += loss.item()
                    val_output = (val_output>0.5).float()
                    dice = dice_metric(val_output, val_mask)

                    #dice_monai =dice_monai_metric(val_output, val_mask)

                    iou = iou_metric(val_output, val_mask)

                    precision = precision_metric(val_output, val_mask)

                    recall = recall_metric(val_output, val_mask)

                    dice_sum += dice
                    iou_sum += iou
                    precision_sum +=precision
                    recall_sum +=recall
                    #dice_monai_sum += dice_monai
                val_dice_mean = dice_sum / val_step
                val_iou_mean = iou_sum / val_step
                val_epoch_loss /= val_step
                precision_mean = precision_sum/val_step
                recall_mean = recall_sum/val_step
                dice_monai_mean = dice_monai_sum/val_step

                print(f" epoch:{i}, train_epoch_loss :{train_epoch_loss}, val_epoch_loss :{val_epoch_loss},dice :{val_dice_mean}, iou{val_iou_mean}, precision:{precision_mean}, recall:{recall_mean}")


                # do something (save model, change lr, etc.)
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
                                  "optimizer":optimizer.state_dict()
                                  }
                    torch.save(checkpoint, file)
                    print('Model saved!')


if __name__ == '__main__':
    args = get_argsparser().parse_args()
    train(args)