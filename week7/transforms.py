# for classification
from torchvision import transforms
classification_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'inf': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop(224),])
}



# for segmentation
from torchvision import transforms
import albumentations
import albumentations.pytorch.transforms as A_torch

def get_transform(transform_ver, mode):
    if mode in ["val", "test"]:
        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
                            ])
        transform_ver = "torchvision"
    elif mode == "train":
        if transform_ver == "albumentation":
            transform = albumentation()
        elif transform_ver == "torchvision":
            transform = torchvision()
    else:
        raise ValueError("Wrong transform mode {}".format(mode))
    return transform, transform_ver

def torchvision():
    transform = transforms.Compose([
                    transforms.ColorJitter(brightness=0.2,
                        contrast=0.4,
                        saturation=0.3,
                        hue=0.25),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                    transforms.AddGaussianNoise(mean=0., std=0.05)
                    ])
    return transform

def albumentation():
    transform = albumentations.Compose([          
                    albumentations.OneOf([
                        albumentations.GaussNoise(),
                    ]),
                    albumentations.OneOf([
                        albumentations.MotionBlur(blur_limit=3, p=0.2),
                        albumentations.MedianBlur(blur_limit=3, p=0.1),
                        albumentations.Blur(blur_limit=2, p=0.1)
                    ]),
                    albumentations.OneOf([
                        albumentations.RandomBrightness(limit=(0.1, 0.4)),
                        albumentations.HueSaturationValue(hue_shift_limit=(0, 128), sat_shift_limit=(0, 60), val_shift_limit=(0, 20)),
                        albumentations.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30)
                    ]),
                    albumentations.OneOf([
                        albumentations.CLAHE(),
                        albumentations.ChannelShuffle(),
                        albumentations.Sharpen(),
                        albumentations.Emboss(),
                        albumentations.RandomBrightnessContrast(),
                    ]),                
                    albumentations.OneOf([
                        albumentations.RandomGamma(gamma_limit=(35,255)),
                        albumentations.OpticalDistortion(),
                        albumentations.GridDistortion(),
                        albumentations.PiecewiseAffine()
                    ]),                
                    albumentations.Normalize(
                        mean= [0.485, 0.456, 0.406],
                        std= [0.229, 0.224, 0.225],
                    ),               
                    A_torch.ToTensorV2()
                    ])
    return transform
