import torch.optim as optim
import time

from einops import layers
from tqdm import tqdm

from models.ee_ucb_resnet_architectures import *
from models.resnet_architectures import *

def train_resnet_static(train_loader, dest, num_epochs, variant):

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  if variant == 50:
    model = ResNetEE50(ResidualBlock50, [3, 4, 3, 6]).to(device)
  else:
    if variant == 34:
        model = ResNetEE18(ResidualBlock, [3, 4, 6, 3]).to(device)
    else:
        model = ResNetEE18(ResidualBlock, [2, 2, 2, 2]).to(device)

  # Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=1e-3)

  loss_weights = [0.9, 0.9, 0.8, 0.7, 0.3]
  criterion = nn.CrossEntropyLoss()

  scaler = torch.cuda.amp.GradScaler()

  start = time.time()

  for epoch in range(num_epochs):
    loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            outputs = model(images)

            losses = [criterion(o, labels) for o in outputs]
            total_loss = sum(w * l for w, l in zip(loss_weights, losses))

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=total_loss.item())

  end = time.time()
  print(f"Elapsed time: {end - start:.4f} seconds")

  # Save the model
  if variant != None:
    variant = str(variant)
    torch.save(model.state_dict(), dest + "/resnet_static_" + str(variant) + ".pth")
    print("Saved with success")