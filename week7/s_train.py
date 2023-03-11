'''
22.02.11. Fri. REAL
~ 22.02.15. Tue.
train: cycle2
test: cycle1, addition
most freq max AP = epoch_35
85.7
88.9
'''

import os

import warnings
warnings.filterwarnings('ignore')

import torch
from MaskRCNN import get_instance_segmentation_model
from s_real_loader import RealTrayDataset
from s_syn_loader import SyntheticDataset
from params import param_segmentation as s_params
from s_coco_utils import get_coco_api_from_dataset, coco_to_excel
from s_engine import collate_fn, train_one_epoch, evaluate


where_train = "real" #"syn"
where_val = "syn" #"real"

# get device (GPU or CPU)
print("... Device Setting")
if torch.cuda.is_available():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# fix seed for reproducibility
torch.manual_seed(7777)

# load dataset
print("... Data loading")
if where_train=="real":
    _dataset = {x: RealTrayDataset(x, s_params) for x in ['train']}
if where_train=="syn":
    _dataset = {x: SyntheticDataset(x, s_params) for x in ['train']}
if where_val=="real":
    _dataset = {x: RealTrayDataset(x, s_params) for x in ['val']}
if where_val=="syn":
    _dataset = {x: SyntheticDataset(x, s_params) for x in ['val']}
_dataloader = {"train": torch.utils.data.DataLoader(dataset=_dataset['train'], num_workers=4,
                                batch_size=2, shuffle=True, collate_fn=collate_fn),
            "val": torch.utils.data.DataLoader(dataset=_dataset['val'], num_workers=4,
                                batch_size=1, shuffle=False, collate_fn=collate_fn)}
print("... Get COCO Dataloader for evaluation")
coco = get_coco_api_from_dataset(_dataloader['val'].dataset)












# load model
print("... Loding MaskRCNN")
num_classes = s_params['num_class']
model = get_instance_segmentation_model(num_classes=num_classes)
model.to(device)

print("... Setting training")
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params)
lr_update = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=3,
                                            gamma=0.1)
# set training epoch    
start_epoch = s_params['start_epoch']
max_epoch = s_params['max_epoch']
assert start_epoch < max_epoch
save_interval = s_params['save_interval']

# logging
output_folder = s_params['save_dir']
os.makedirs(output_folder, exist_ok=True)
print("Model save directory: " + output_folder)
print("="*20)
print("+++ Start Training  @start:{} @max: {}".format(start_epoch, max_epoch))
for epoch in range(start_epoch, max_epoch):
    # train
    train_one_epoch(epoch, model, _dataloader["train"], optimizer, device, lr_update)
    # validate and write results
    coco_evaluator = evaluate(coco, model, _dataloader["val"], device)
    # save weight
    if epoch % save_interval == 0:
        torch.save(model.state_dict(), '{}/epoch_{}.tar'.format(output_folder, epoch))
        coco_to_excel(
            coco_evaluator, epoch, output_folder, 
            "{}_{}".format("real_tray", "unseen_food"))


