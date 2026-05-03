import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATASET_PATH  = "dataset"
BATCH_SIZE    = 4        
ACCUM_STEPS   = 4         
EPOCHS        = 5         
LR            = 7e-5       
KEEP_TOKENS   = 80        
PRUNE_AFTER   = 6          
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device       : {DEVICE}")
print(f"Effective batch    : {BATCH_SIZE * ACCUM_STEPS} (accum {ACCUM_STEPS} steps)")

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(
    root=os.path.join(DATASET_PATH, "train"),
    transform=train_tf
)
test_dataset = datasets.ImageFolder(
    root=os.path.join(DATASET_PATH, "test"),
    transform=test_tf
)

PIN = DEVICE == "cuda"
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0, pin_memory=PIN
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0, pin_memory=PIN
)

print(f"Train samples : {len(train_dataset)}")
print(f"Test  samples : {len(test_dataset)}")
print(f"Class mapping : {train_dataset.class_to_idx}")

class LatePrunedViT(nn.Module):

    def __init__(self, keep_tokens=80, prune_after=6):
        super().__init__()
        self.vit         = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.keep_tokens = keep_tokens
        self.prune_after = prune_after

        self.vit.heads = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.vit._process_input(x)
        n = x.shape[0]

        cls_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.encoder.pos_embedding

        for layer in self.vit.encoder.layers[:self.prune_after]:
            x = layer(x)

        last_layer_name = f"encoder_layer_{self.prune_after - 1}"
        last_layer = self.vit.encoder.layers._modules[last_layer_name]

        with torch.no_grad():
            _, attn_weights = last_layer.self_attention(
                x, x, x,
                need_weights=True,
                average_attn_weights=True 
            )
        scores    = attn_weights[:, 0, 1:]                          
        topk      = torch.topk(scores, self.keep_tokens, dim=1).indices 
        batch_idx = torch.arange(n, device=x.device).unsqueeze(-1)

        cls     = x[:, 0:1, :]
        patches = x[:, 1:,  :]
        patches = patches[batch_idx, topk]         
        x       = torch.cat((cls, patches), dim=1)  

        for layer in self.vit.encoder.layers[self.prune_after:]:
            x = layer(x)

        x          = self.vit.encoder.ln(x)
        cls_output = x[:, 0]
        return self.vit.heads(cls_output)

model = LatePrunedViT(keep_tokens=KEEP_TOKENS, prune_after=PRUNE_AFTER)

for name, param in model.named_parameters():
    if (
        "encoder_layer_9"  in name or
        "encoder_layer_10" in name or
        "encoder_layer_11" in name or
        "encoder.ln"       in name or
        "heads"            in name
    ):
        param.requires_grad = True
    else:
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params : {trainable:,} / {total:,}")

if DEVICE == "cuda":
    torch.cuda.empty_cache()

model.to(DEVICE)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=0.01
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS
)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

def train_one_epoch(model, loader):
    model.train()
    losses = []
    optimizer.zero_grad()

    for step, (x, y) in enumerate(loader):
        x, y    = x.to(DEVICE), y.to(DEVICE)
        outputs = model(x)
        loss    = criterion(outputs, y) / ACCUM_STEPS
        loss.backward()
        losses.append(loss.item() * ACCUM_STEPS)  

        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

    return np.mean(losses)

def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x     = x.to(DEVICE)
            preds = torch.argmax(model(x), dim=1).cpu()
            y_true.extend(y.numpy())
            y_pred.extend(preds.numpy())

    acc  = accuracy_score (y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score   (y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score       (y_true, y_pred, average='macro', zero_division=0)

    return acc, prec, rec, f1

best_f1     = 0.0
no_improve  = 0
epoch_times = []

print("\n" + "="*55)
print(f"  Training  |  keep_tokens={KEEP_TOKENS}  prune_after=layer {PRUNE_AFTER}")
print("="*55)

for epoch in range(EPOCHS):
    start_time = time.time()

    train_loss         = train_one_epoch(model, train_loader)
    acc, prec, rec, f1 = evaluate(model, test_loader)

    scheduler.step()

    epoch_time = (time.time() - start_time) / 60
    epoch_times.append(epoch_time)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss : {train_loss:.4f}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1-score   : {f1:.4f}")
    print(f"  LR         : {scheduler.get_last_lr()[0]:.2e}")
    print(f"  Time       : {epoch_time:.2f} min")

    if f1 > best_f1:
        best_f1    = f1
        no_improve = 0
        torch.save(model.state_dict(), "best_vit_deepfake.pth")
        print("  Best model saved.")

total_time = sum(epoch_times)
print("\n" + "="*55)
print(f"  Best F1-score            : {best_f1:.4f}")
print(f"  Avg time per epoch       : {total_time/len(epoch_times):.2f} min")
print(f"  Total training time      : {total_time:.2f} min")
print(f"  Tokens kept per forward  : {KEEP_TOKENS+1} / 197")
print(f"  Token reduction          : {((197-(KEEP_TOKENS+1))/197)*100:.1f}%")
print("="*55)
print("Training complete.")