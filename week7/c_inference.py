import torch
import os
from torchvision import datasets
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import font_manager, rc

import c_forInf as CI
from params import param_food as params_setting
from params import food_id as class_to_name
from transforms import classification_transforms as data_transform



data_dir = params_setting['food_root']
dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transform['train'])

print("+++ Load model checkpoint")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = CI.load_checkpoint(device, dataset, params_setting)
model.eval()

inf_path = os.path.join(params_setting['food_root'], 'inf') 
img_list = sorted(os.listdir(inf_path))
for i, img_name in enumerate(img_list):
    save_path = os.path.join(params_setting['graph_path'],img_name) 
    img_file = os.path.join(inf_path, img_name) # 여기보는중
    probs, classes = CI.predict1(img_file, model) # 여기보는중
    print(img_name)  
    print(probs)
    print(classes)

    # Display an image along with the top 5 classes
    # Plot flower input image
    plt.figure(figsize = (6,10))
    plot_1 = plt.subplot(2,1,1)
    image = CI.process_image(img_file)
    CI.imshow(image, plot_1, title=img_name);
    # Convert from the class integer encoding to actual flower names
    image_names = [i for i in classes]
    # Plot the probabilities for the top 5 classes as a bar graph
    plt.subplot(2,1,2)
    sb.barplot(x=probs, y=image_names, color=sb.color_palette()[0]);

    plt.savefig(save_path)
