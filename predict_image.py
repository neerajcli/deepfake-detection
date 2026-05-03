import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.models import vit_b_16
from PIL import Image

MODEL_PATH           = "best_vit_deepfake.pth"
IMAGE_SIZE           = 224
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES          = ["fake", "real"]
CONFIDENCE_THRESHOLD = 0.5

class LatePrunedViT(nn.Module):
    def __init__(self, keep_tokens=80, prune_after=6):
        super().__init__()
        self.vit         = vit_b_16(weights=None)
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
        last_layer      = self.vit.encoder.layers._modules[last_layer_name]

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

class ImagePreprocessor:
    def __init__(self, image_size):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess(self, image_path):
        image  = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor.unsqueeze(0)

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = LatePrunedViT(keep_tokens=80, prune_after=6)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def predict_image(image_path, model, preprocessor):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_tensor = preprocessor.preprocess(image_path).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    idx = int(np.argmax(probs))

    return {
        "label"        : CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "confident": float(probs[idx]) > CONFIDENCE_THRESHOLD,
        "probabilities": {
            CLASS_NAMES[0]: float(probs[0]),
            CLASS_NAMES[1]: float(probs[1])
        }
    }

def main():
    image_path   = input("Enter image path: ").strip()
    preprocessor = ImagePreprocessor(IMAGE_SIZE)
    model        = load_model(MODEL_PATH)
    result       = predict_image(image_path, model, preprocessor)

    print("\nPrediction Result")
    print("------------------")
    print(f"Predicted Class : {result['label'].upper()}")
    print(f"Confidence      : {result['confidence'] * 100:.2f}%")
    print(f"Confident       : {result['confident']}")
    print("Class Probabilities:")
    for cls, prob in result["probabilities"].items():
        print(f"  {cls}: {prob * 100:.2f}%")

if __name__ == "__main__":
    main()