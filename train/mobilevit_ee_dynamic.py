from tqdm import tqdm
from models.mobilevit_architectures import *
from models.ee_mobilevit_architectures import *
from models.ee_ucb_resnet_architectures import *

def train_mvit_dynamic(train_loader, dest, src, num_epochs=20, classes=10):

    pretrained_model = mobilevit_xxs()
    pretrained_model.load_state_dict(torch.load(src))

    model = MobileViTWithEarlyExits(pretrained_model, exit_points=['mvit_0', 'mvit_1'], num_classes=classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gating_model = Gmodel(hidden_dim=10).to(device)
    global_eval = [torch.tensor(0.0, device=device) for _ in range(4)]
    global_count = [torch.tensor(0.0, device=device) for _ in range(4)]

    global_coverage_eval = [torch.tensor(0.0, device=device) for _ in range(4)]
    global_coverage_count = [torch.tensor(0.0, device=device) for _ in range(4)]

    scaler = torch.cuda.amp.GradScaler()

    for param in model.base.parameters():
       param.requires_grad = False

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(gating_model.parameters()), lr=5e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = torch.nn.CrossEntropyLoss()

    print("\n=== Training early exits only (frozen backbone) ===")
    for epoch in range(num_epochs):
        model.to(device)
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Phase1 Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
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

            second_term = [l*d + (max(0, c-g)**2) for l, d, c, g in zip(first_term, pen, c_term, phi_term)]

            denominator = [1, 2, 3]

            total_loss = sum([(a+b)*c for a,b,c in zip(first_term, second_term, denominator)])
            denominator_sum = sum(denominator)
            total_loss = total_loss / denominator_sum

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f"[Phase1] Epoch {epoch+1}: Train Loss={total_loss:.4f}")

        scheduler.step()

    torch.save(model.state_dict(), dest + "/mobilevit_dynamic" + str(classes) + ".pth")
    torch.save(gating_model.state_dict(), dest + "/gating" + str(classes) + ".pth")

    return