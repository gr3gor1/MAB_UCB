import time

from tqdm import tqdm
from models.resnet_architectures import *

def train_resnet(train_loader, dest, num_epochs, variant):

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Model Initialization
  if variant == 50:
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

  # Save the model
  if variant is not None:
    variant = str(variant)
    torch.save(model.state_dict(), dest + "/resnet_" + str(variant) + ".pth")
    print("Saved with success")

  return