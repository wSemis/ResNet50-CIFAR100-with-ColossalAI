import argparse

import torch
import torchvision
import torchvision.transforms as transforms

import re
import os
import json
# ==============================
# Parse Arguments
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--checkpoint", type=str, default="./checkpoint", help="checkpoint directory")
args = parser.parse_args()

# ==============================
# Prepare Test Dataset
# ==============================
# CIFAR-10 dataset
test_dataset = torchvision.datasets.CIFAR100(root="./data/", train=False, transform=transforms.ToTensor())

# Data loader
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

# ==============================
# Load Model
# ==============================
model = torchvision.models.resnet50(num_classes=100).cuda()

# Get all pth files
pattern = re.compile(r'model_\d+\.pth$')

matched_files = []

for filename in os.listdir(args.checkpoint):
    if pattern.match(filename):
        matched_files.append(filename)

accuracies = []
epochs = []
for pth in matched_files:
    epoch = int(pth.split('_')[1].split('.')[0])
    state_dict = torch.load(f"{args.checkpoint}/{pth}")
    model.load_state_dict(state_dict)

    # ==============================
    # Run Evaluation
    # ==============================
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Epoch {}, Accuracy: {} %, correct: {}, total: {}".format(epoch, 100 * correct / total, correct, total))
    accuracies.append(100 * correct / total)
    epochs.append(epoch)
json.dump({'epochs': epochs, 'accuracies': accuracies}, open('accuracies.json', 'w'))