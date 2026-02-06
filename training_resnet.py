import torch
import torch.nn as nn
import time

from tqdm import tqdm
from resnet_architectures import *

def training(train_loader, valid_loader, dest, itsfifty=False, variant=None):

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Model Initialization
  if itsfifty:
    model = ResNet50(ResidualBlock50, [3, 4, 6, 3]).to(device)
  else:
    if variant == 34:
        model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)
    else:
        model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)

  # Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.001, momentum = 0.9)

  start = time.time()

  scaler = torch.cuda.amp.GradScaler()

  num_epochs = 10

  for epoch in range(num_epochs):
    loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

  end = time.time()

  print(f"Elapsed time: {end - start:.4f} seconds")


  # Validation
  with torch.no_grad():
      correct = 0
      total = 0
      for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

  print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))

  # Save the model
  if variant is not None:
    variant = str(variant)
    torch.save(model.state_dict(), dest + "/resnet_" + str(variant) + ".pth")
    print("Saved with success")

  return