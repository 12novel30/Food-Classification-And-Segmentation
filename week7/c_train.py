import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets
import os
from torch.utils.data import DataLoader
from transforms import classification_transforms as data_transform
from params import param_classification as c_params
from ResNet import get_resnet50_from
import c_forTrain as C

'''
0216_10:30
resnet50으로 재학습
'''
print("~ Start Food Classification! ~")
data_dir = c_params['food_root']
dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transform[x])
            for x in ['train', 'val']}
dataloader = {x: DataLoader(dataset[x], c_params['batch_size'], shuffle=True)
               for x in ['train', 'val']}

dataset_sizes = {x: len(dataset[x]) for x in ['train', 'val']}
class_names = dataset['train'].classes
num_classes = c_params['food_classes']
if num_classes != len(class_names):
    print("#(Class) ERROR!!")
print("... finish data setting")

#device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda")
if torch.cuda.is_available():
    print("cuda is: available")
else:
    print("ERROR!")
print("... finish cuda setting")
model = get_resnet50_from(device, num_classes)
print("... finish ResNet50 setting")
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

params_train = {
    'num_epochs':c_params['num_epochs'],
    'optimizer':optimizer_ft,
    'dataloader':dataloader,
    'dataset_sizes':dataset_sizes,
    'device':device,
    'lr_scheduler':exp_lr_scheduler,
    'model_folder':os.path.join(c_params['model_folder'], '0218_04:49'),
    'path2weights':os.path.join(c_params['model_folder'], c_params['weight_']),
    'criterion':criterion
}
C.createFolder(params_train['model_folder'])
print("... finish training setting")
print("="*20)

print("+++ Start Training  @max: {}".format(c_params['num_epochs']))
model, loss_hist, metric_hist = C.train_model(model, params_train)


print("+++ Show loss and accuracy graph")
C.get_loss_graph(c_params['num_epochs'], loss_hist, c_params['graph_path']+'18_04:50/')
# C.graph_reset()
C.get_accuracy_graph(c_params['num_epochs'], metric_hist, c_params['graph_path']+'18_04:50/')
