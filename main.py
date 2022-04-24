
import sys
import torch
import logging
import argparse
import torchmetrics
import torch.nn as nn
from torch import optim
from datetime import datetime
torch.backends.cudnn.benchmark = True

import commons
from models import build_network
from loss import CrossEntropyLabelSmooth
from datasets import build_dataset, Augment

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch_size", type=int, default=500, help="_")
parser.add_argument("--num_workers", type=int, default=3, help="_")
parser.add_argument("--epochs_num", type=int, default=25, help="_")
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_")
parser.add_argument("--seed_weights", type=int, default=0, help="_")
parser.add_argument("--seed_optimization", type=int, default=0, help="_")

args = parser.parse_args()
start_time = datetime.now()
output_folder = f"logs/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
# output_folder = f"logs/sw_{args.seed_weights:02d}__so_{args.seed_optimization:02d}"
commons.make_deterministic(args.seed_optimization)
commons.setup_logging(output_folder, console="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {output_folder}")

train_imgs, train_labels, val_imgs, val_labels, test_imgs, test_labels = build_dataset(device=args.device)

import datasets
val_dataset = datasets.CustomDataset(val_imgs, val_labels)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
test_dataset = datasets.CustomDataset(test_imgs, test_labels)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

n_train = len(train_labels)
n_val = len(val_labels)
n_test = len(test_labels)

net = build_network()
net.load_state_dict(torch.load(f"weights/{args.seed_weights:02d}.pth"))
net = net.to(args.device).half()

criterion = nn.CrossEntropyLoss()
criterion2 = CrossEntropyLabelSmooth(num_classes=10, epsilon=0.2)
optimizer = optim.SGD(net.parameters(), lr=0.2, momentum=0.9, nesterov=True, weight_decay=0.001)

def lr(e):
    if e < 4:
        return 0.5*e/3. + 0.01
    elif e < 22:
        return 0.5*(22-e)/19. + 0.01
    return 0.01

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr)

augment = Augment()
augment = augment.to(args.device).half()

best_val_accuracy = 0

for epoch_num in range(args.epochs_num):
    train_loss = torchmetrics.MeanMetric()
    
    # process training set
    augm_train = []
    for i in range(n_train // args.batch_size):
        # get the inputs
        inputs = train_imgs[i*args.batch_size:(i+1)*args.batch_size, ...]
        augm_train.append(augment(inputs.to(args.device).half()))
    augm_train_imgs = torch.cat(augm_train)
    perm = torch.randperm(n_train)
    augm_train_imgs = augm_train_imgs[perm, ...].contiguous()
    augm_train_labels = train_labels[perm].contiguous()
    train_dataset = datasets.CustomDataset(augm_train_imgs, augm_train_labels)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    
    net = net.train()
    for inputs, labels in train_dl:
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss2 = criterion2(outputs, labels)
        loss = loss + 2*loss2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        train_loss.update(loss.item())
    scheduler.step()
    
    #### Validation
    val_accuracy = torchmetrics.Accuracy().cuda()
    val_loss = torchmetrics.MeanMetric()
    net = net.eval()
    with torch.no_grad():
        for inputs, labels in val_dl:
            outputs = net(inputs)
            val_accuracy.update(outputs, labels)
            val_loss.update(criterion(outputs, labels).item())
    
    val_accuracy = val_accuracy.compute().item() * 100
    if val_accuracy > best_val_accuracy:
        torch.save(net.state_dict(), f"{output_folder}/best_model.pth")
        best_val_accuracy = val_accuracy
    
    logging.debug(f"Epoch: {epoch_num + 1:02d}/{args.epochs_num}; " +
                  f"train_loss: {train_loss.compute().item():.3f}; " +
                  f"val_loss: {val_loss.compute().item():.3f}; " +
                  f"val_accuracy: {val_accuracy:.2f}%; " +
                  f"best_val_accuracy: {best_val_accuracy:.2f}%")

#### Test with best model
net.load_state_dict(torch.load(f"{output_folder}/best_model.pth"))
net = net.eval()

test_accuracy = torchmetrics.Accuracy().cuda()
with torch.no_grad():
    test_accuracy = torchmetrics.Accuracy().cuda()
    test_loss = torchmetrics.MeanMetric()
    net = net.eval()
    for inputs, labels in test_dl:
        outputs = net(inputs)
        test_accuracy.update(outputs, labels)
        test_loss.update(criterion(outputs, labels).item())

test_accuracy = test_accuracy.compute().item() * 100
logging.info(f"Training took {str(datetime.now() - start_time)[:-7]}; " +
             f"best_val_accuracy: {best_val_accuracy:.2f}; " +
             f"test_accuracy: {test_accuracy:.2f}")

