from tqdm import tqdm
from models.mobilevit_architectures import *
from models.ee_mobilevit_architectures import *

def validate(model, val_loader, device, exit_weights=[0.9, 0.7, 0.5]):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)

            # Forward pass for losses
            losses = model(images, targets, train_mode=True)
            batch_loss = sum(w * l for w, l in zip(exit_weights, losses)) * 0.33
            total_loss += batch_loss.item()

            # Forward pass in inference mode
            _, preds = model(images, train_mode=False)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def train_mvit_static(train_loader, valid_loader, dest, num_epochs, src, classes=10):

    pretrained_model = mobilevit_xxs()
    pretrained_model.load_state_dict(torch.load(src))

    model = MobileViTWithEarlyExits(pretrained_model, exit_points=['mvit_0', 'mvit_1'], num_classes=classes)

    scaler = torch.cuda.amp.GradScaler()

    for param in model.base.parameters():
        param.requires_grad = False  # freeze backbone

    optimizer = torch.optim.AdamW(model.exits.parameters(), lr=5e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    best_val_loss = float('inf')
    patience = 7
    epochs_no_improve = 0
    exit_weights = [0.9, 0.7, 0.5]

    print("\n=== Training early exits only (frozen backbone) ===")
    for epoch in range(num_epochs):
        model.to(device)
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Phase1 Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                losses = model(images, labels, train_mode=True)
                # Use only exit losses, skip final classifier loss
                loss = sum(w * l for w, l in zip(exit_weights, losses)) * 0.33

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = validate(model, valid_loader, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")

        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), dest + "/mobilevit_static_" + str(classes) + ".pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Phase1] Early stopping after {epoch+1} epochs.")
                break