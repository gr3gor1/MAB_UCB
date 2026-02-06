import torch.optim as optim
import time

from tqdm import tqdm

from ee_resnet_architectures import *
from resnet_architectures import *

def train_offline_ee(epochs, dataloader, variant, dest):

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  gating_model = Gmodel(hidden_dim=10).to(device)

  if variant == 50:
    model = ResNetEE50(ResidualBlock50, [3, 4, 6, 3]).to(device)
  else:
    if variant == 34:
        model = ResNetEE18(ResidualBlock, [3, 4, 6, 3]).to(device)
    else:
        model = ResNetEE18(ResidualBlock, [2, 2, 2, 2]).to(device)

  global_eval = [torch.tensor(0.0, device=device) for _ in range(4)]
  global_count = [torch.tensor(0.0, device=device) for _ in range(4)]

  global_coverage_eval = [torch.tensor(0.0, device=device) for _ in range(4)]
  global_coverage_count = [torch.tensor(0.0, device=device) for _ in range(4)]

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(list(model.parameters()) + list(gating_model.parameters()), lr=1e-3)
  scaler = torch.cuda.amp.GradScaler()

  start = time.time()

  for epoch in range(epochs):
    loop = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")

    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            outputs = model(images)

            losses = [criterion(o, labels) for o in outputs]
            pen = [gating_model(o).mean() for o in outputs]
            first_term = losses

            second_term = []

            c_term = [0, 0, 0, 0]
            phi_term = []

            for idx, o in enumerate(outputs):

              check = gating_model(o) >= 0.5
              global_eval[idx] += check.sum()
              global_count[idx] += check.numel()


              phi_term.append(global_eval[idx] / global_count[idx])

              if epoch > 1:
                pred = torch.argmax(o, dim=1)
                hit = (pred == labels).sum()

                global_coverage_eval[idx] += hit
                global_coverage_count[idx] += labels.numel()

                c_term[idx] = global_coverage_eval[idx] / global_coverage_count[idx]

            second_term = [l*d + (max(0, c-g)**2) for l, d, c, g in zip(losses, pen, c_term, phi_term)]

            denominator = [1, 2, 3, 4]

            total_loss = sum([(a+b)*c for a,b,c in zip(first_term, second_term, denominator)])
            denominator_sum = sum(denominator)
            total_loss = total_loss / denominator_sum

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=total_loss.item())

  end = time.time()

  torch.save(model.state_dict(), dest + "/ee_resnet_model_" + str(variant) + ".pth")
  torch.save(gating_model.state_dict(), dest + "/gating_" + str(variant) + "_weights.pth")
  print(f"Elapsed time: {end - start:.4f} seconds")

  return
