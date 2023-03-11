from torchvision import models
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

from transforms import classification_transforms as data_transform


def load_checkpoint(device, training_dataset, params):
    filepath = params['model_weights_path']
    num_classes = params['food_classes']
    model = models.resnet18(pretrained=True)
    model.class_to_idx = training_dataset.class_to_idx
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.to(device)
    model.load_state_dict(torch.load(filepath))
    
    return model

def load_checkpoint_real(device, params): #임시로 저장해두기...
    filepath = params['checkpoint_path']
    num_classes = params['food_classes']
    
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'ResNet18':
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Architecture not recognized.")
    
    model.class_to_idx = checkpoint['class_to_idx']
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


# in [predict]
def process_image(image_path):
    img_data = Image.open(image_path) 
    img_data = data_transform['inf'](img_data)
    np_image = np.array(img_data)/255 # 여기보는중
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    if title is not None:
        ax.set_title(title)
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    
    return ax

def predict5(image_path, model, topk=5):
    image = process_image(image_path)
    # Convert image to PyTorch tensor first
    image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    image = image.unsqueeze(0)
    
    output = model.forward(image)
    probabilities = torch.exp(output)
    # Probabilities and the indices of those probabilities corresponding to the classes
    top_probabilities, top_indices = probabilities.topk(topk)
    # Convert to lists
    top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    
    # Convert topk_indices to the actual class labels using class_to_idx
    # Invert the dictionary so you get a mapping from index to class.
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    top_classes = [idx_to_class[index] for index in top_indices]
    
    return top_probabilities, top_classes

def predict1(image_path, model, topk=1):
    image = process_image(image_path) 
    # Convert image to PyTorch tensor first
    image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    image = image.unsqueeze(0) # 여기보는중
    
    output = model.forward(image)
    probabilities = torch.exp(output)
    # Probabilities and the indices of those probabilities corresponding to the classes
    top_probabilities, top_indices = probabilities.topk(topk)
    # Convert to lists
    top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    
    # Convert topk_indices to the actual class labels using class_to_idx
    # Invert the dictionary so you get a mapping from index to class.
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    top_classes = [idx_to_class[index] for index in top_indices]
    
    return top_probabilities, top_classes


