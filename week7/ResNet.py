from torchvision import models
import torch.nn as nn
import torch
from torchsummary import summary 

def get_resnet18_from(device, data_classes):
    resnet18_from = models.resnet18(pretrained=True)
    num_ftrs = resnet18_from.fc.in_features
    resnet18_from.fc = nn.Linear(num_ftrs, data_classes)
    resnet18_from = resnet18_from.to(device)
    return resnet18_from

def get_resnet50_from(device, data_classes):
    resnet50_from = models.resnet50(pretrained=True)
    num_ftrs = resnet50_from.fc.in_features
    resnet50_from.fc = nn.Linear(num_ftrs, data_classes)
    resnet50_from = resnet50_from.to(device)
    return resnet50_from

def get_model_output_size(model, device):
    model = model.to(device)
    x = torch.randn(3, 3, 224, 224).to(device)
    output = model(x)
    print("model; output size: ")
    print(output.size())

def get_model_summary(model, device):
    print("summary: ")
    summary(model, (3,224,224), device=device.type)