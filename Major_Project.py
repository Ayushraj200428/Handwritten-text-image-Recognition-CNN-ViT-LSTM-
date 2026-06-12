from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import timm
from pathlib import Path
from io import BytesIO
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# ============================================================================
# CHARACTER SET
# ============================================================================
DEVANAGARI_CHARS = [
    '-', 'ँ', 'ं', 'ः', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ऌ', 'ऍ', 'ऎ',
    'ए', 'ऐ', 'ऑ', 'ऒ', 'ओ', 'औ', 'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज',
    'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'ऩ', 'प',
    'फ', 'ब', 'भ', 'म', 'य', 'र', 'ऱ', 'ल', 'ळ', 'ऴ', 'व', 'श', 'ष', 'स',
    'ह', '़', 'ऽ', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॅ', 'े', 'ै', 'ॉ',
    'ॊ', 'ो', 'ौ', '्', 'ॐ', '॑', '॒', '॓', '॔', 'क़', 'ख़', 'ग़', 'ज़', 'ड़',
    'ढ़', 'फ़', 'य़', 'ॠ', 'ॡ', '।', '॥', '०', '१', '२', '३', '४', '५', '६',
    '७', '८', '९', '॰', 'ॱ', 'ॲ', 'ॻ', 'ॼ', 'ॽ', 'ॾ',
    '<BLANK>'
]

BLANK_IDX   = len(DEVANAGARI_CHARS) - 1   # 108
NUM_CLASSES = len(DEVANAGARI_CHARS)        # 109

# ============================================================================
# SHARED PREPROCESSING
# Pipeline: grayscale -> CLAHE -> aspect-ratio resize with white padding -> /255
# Dark strokes on light background, identical for training and inference.
# ============================================================================
def preprocess_image(image_input, target_size=(64, 256)):
    H, W = target_size
    if isinstance(image_input, Image.Image):
        img = np.array(image_input.convert('L'), dtype=np.uint8)
    elif image_input.ndim == 3:
        img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
    else:
        img = image_input.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)

    ih, iw  = img.shape
    scale   = min(W / iw, H / ih)
    nw, nh  = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas          = np.full((H, W), 255, dtype=np.uint8)
    y_off, x_off    = (H - nh) // 2, (W - nw) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    return canvas.astype(np.float32) / 255.0

# ============================================================================
# PART 1: CNN FEATURE EXTRACTOR
# Input (1, H, W) -> (feature_dim, H/4, W/4) after two MaxPool2d(2,2).
# With TARGET_SIZE=(64,256): output spatial = (16, 64).
# ============================================================================
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, feature_dim=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim), nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))

# ============================================================================
# PART 2: TOKEN EMBEDDING
# patch_size=8 on CNN features (spatial 16x64) gives 2x8=16 tokens per axis
# -> 16*8=128 tokens total. Larger patch captures broader stroke context.
# ============================================================================
class TokenEmbedding(nn.Module):
    def __init__(self, feature_dim=256, embed_dim=384, patch_size=8):
        super().__init__()
        self.projection = nn.Conv2d(feature_dim, embed_dim,
                                    kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x        = self.projection(x)
        _, _, H, W = x.shape
        x        = x.flatten(2).transpose(1, 2)   # (B, N, embed_dim)
        return x, (H, W)

# ============================================================================
# PART 3: POSITIONAL ENCODING
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len=1000, embed_dim=384):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02)

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]

# ============================================================================
# PART 4: VISION TRANSFORMER ENCODER
# ============================================================================
class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=384, depth=6, num_heads=6, pretrained=False):
        super().__init__()
        if pretrained:
            vit = timm.create_model('vit_small_patch16_224',
                                    pretrained=True, num_classes=0, global_pool='')
            self.transformer_blocks = vit.blocks
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1, activation='gelu', batch_first=True)
            self.transformer_blocks = nn.TransformerEncoder(
                encoder_layer, num_layers=depth)

    def forward(self, x):
        return self.transformer_blocks(x)

# ============================================================================
# PART 5: BiLSTM SEQUENCE MODELER
# Inserted AFTER ViT encoder, BEFORE CTC head.
# Forward LSTM captures left context; backward LSTM captures right context.
# Architecture: embed_dim -> hidden_dim*2 (bidirectional) -> embed_dim (proj)
# ============================================================================
class BiLSTMSequenceModeler(nn.Module):
    def __init__(self, embed_dim=384, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0)
        self.proj       = nn.Linear(hidden_dim * 2, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.bilstm(x)          # (B, T, hidden_dim*2)
        out     = self.proj(out)          # (B, T, embed_dim)
        out     = self.dropout(out)
        return self.layer_norm(out + x)  # residual connection

# ============================================================================
# PART 6: CTC PREDICTION HEAD
# ============================================================================
class CTCPredictionHead(nn.Module):
    def __init__(self, embed_dim=384, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes))

    def forward(self, x):
        return self.fc(x) 

# ============================================================================
# PART 7: COMPLETE HYBRID MODEL  CNN -> ViT -> BiLSTM -> CTC
# ============================================================================
class HybridCNNViT_HindiOCR(nn.Module):
    def __init__(self, input_channels=1, feature_dim=256, embed_dim=384,
                 vit_depth=6, vit_heads=6, num_classes=NUM_CLASSES,
                 patch_size=8, bilstm_hidden=256, bilstm_layers=2,
                 pretrained_vit=False, char_list=None):
        super().__init__()
        self.cnn_backbone    = CNNFeatureExtractor(input_channels, feature_dim)
        self.token_embedding = TokenEmbedding(feature_dim, embed_dim, patch_size)
        self.pos_encoding    = PositionalEncoding(1000, embed_dim)
        self.vit_encoder     = ViTEncoder(embed_dim, vit_depth, vit_heads, pretrained_vit)
        self.bilstm          = BiLSTMSequenceModeler(embed_dim, bilstm_hidden, bilstm_layers) if bilstm_layers > 0 else None
        self.prediction_head = CTCPredictionHead(embed_dim, num_classes)
        self.char_list       = char_list if char_list is not None else DEVANAGARI_CHARS

    def forward(self, x, return_features=False):
        cnn_feat          = self.cnn_backbone(x)
        tokens, spatial   = self.token_embedding(cnn_feat)
        tokens            = self.pos_encoding(tokens)
        vit_feat          = self.vit_encoder(tokens)
        seq_feat          = self.bilstm(vit_feat) if self.bilstm is not None else vit_feat
        logits            = self.prediction_head(seq_feat)
        if return_features:
            return logits, {
                'cnn_features': cnn_feat, 'tokens': tokens,
                'vit_features': vit_feat, 'seq_features': seq_feat,
                'spatial_shape': spatial}
        return logits

    def decode_predictions(self, logits):
        log_probs = F.log_softmax(logits, dim=-1)
        preds     = torch.argmax(log_probs, dim=-1)
        decoded_texts, decoded_confs = [], []
        for b in range(preds.size(0)):
            seq    = preds[b].cpu().numpy()
            lp_seq = log_probs[b].cpu().numpy()
            chars, lp_chars = [], []
            prev = None
            for t, idx in enumerate(seq):
                if idx != BLANK_IDX and idx != prev:
                    chars.append(self.char_list[idx])
                    lp_chars.append(lp_seq[t, idx])
                prev = idx
            decoded_texts.append(''.join(chars))
            decoded_confs.append(
                float(np.exp(np.mean(lp_chars))) if lp_chars else 0.0)
        return decoded_texts, decoded_confs

# ============================================================================
# PART 8: AUGMENTATION
# ============================================================================
def augment_image(image: np.ndarray) -> np.ndarray:
    """Apply random augmentations to a uint8 grayscale image."""
    h, w = image.shape

    # 1. Rotation
    if np.random.rand() < 0.5:
        angle = np.random.uniform(-10, 10)
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 2. Horizontal shear
    if np.random.rand() < 0.4:
        shear   = np.random.uniform(-0.15, 0.15)
        M_shear = np.float32([[1, shear, 0], [0, 1, 0]])
        image   = cv2.warpAffine(image, M_shear, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 3. Elastic distortion
    if np.random.rand() < 0.4:
        alpha = np.random.uniform(4, 10)
        sigma = np.random.uniform(3, 5)
        dx    = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32) * alpha, (0, 0), sigma)
        dy    = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32) * alpha, (0, 0), sigma)
        grid_x = np.meshgrid(np.arange(w), np.arange(h))[0].astype(np.float32) + dx
        grid_y = np.meshgrid(np.arange(w), np.arange(h))[1].astype(np.float32) + dy
        image  = cv2.remap(image, grid_x, grid_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)

    # 4. Morphological dilation / erosion
    if np.random.rand() < 0.4:
        ksize  = np.random.choice([2, 3])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        if np.random.rand() < 0.5:
            image = cv2.dilate(image, kernel)
        else:
            image = cv2.erode(image, kernel)

    # 5. Blur
    if np.random.rand() < 0.3:
        if np.random.rand() < 0.5:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        else:
            image = cv2.medianBlur(image, 3)

    # 6. Brightness + contrast jitter
    if np.random.rand() < 0.5:
        alpha = np.random.uniform(0.7, 1.3)
        beta  = np.random.randint(-20, 20)
        image = np.clip(image.astype(np.int16) * alpha + beta, 0, 255).astype(np.uint8)

    # 7. Gaussian noise
    if np.random.rand() < 0.3:
        noise = np.random.normal(0, np.random.uniform(3, 8), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 8. Random perspective warp
    if np.random.rand() < 0.3:
        margin  = int(min(h, w) * 0.06)
        src     = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst     = np.float32([
            [np.random.randint(0, margin), np.random.randint(0, margin)],
            [w - np.random.randint(0, margin), np.random.randint(0, margin)],
            [w - np.random.randint(0, margin), h - np.random.randint(0, margin)],
            [np.random.randint(0, margin), h - np.random.randint(0, margin)]])
        M_persp = cv2.getPerspectiveTransform(src, dst)
        image   = cv2.warpPerspective(image, M_persp, (w, h),
                                      borderMode=cv2.BORDER_REPLICATE)
    return image

# ============================================================================
# PART 9: PARQUET DATASET
# ============================================================================
class ParquetHandwritingDataset(Dataset):
    def __init__(self, parquet_path, char_list, target_size=(64, 256),
                 augment=False, max_samples=None):
        self.char_list   = char_list
        self.char_to_idx = {c: i for i, c in enumerate(char_list)}
        self.target_size = target_size
        self.augment     = augment

        print(f"Loading {parquet_path} ...")
        df = pd.read_parquet(parquet_path)

        def all_known(text):
            return all(c in self.char_to_idx for c in str(text))

        df = df[df['text'].apply(all_known)].reset_index(drop=True)
        if max_samples is not None:
            df = df.sample(n=min(max_samples, len(df)),
                           random_state=42).reset_index(drop=True)

        self.image_bytes = df['image'].tolist()
        self.texts       = df['text'].tolist()
        print(f"  -> {len(self.texts)} usable samples")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        raw       = self.image_bytes[idx]
        img_bytes = raw['bytes'] if isinstance(raw, dict) else raw
        pil_img   = Image.open(BytesIO(img_bytes)).convert('L')
        img_np    = np.array(pil_img, dtype=np.uint8)

        if self.augment:
            img_np  = augment_image(img_np)
            pil_img = Image.fromarray(img_np)

        image        = preprocess_image(pil_img, self.target_size)
        image_tensor = torch.from_numpy(image).unsqueeze(0)

        label_indices = [self.char_to_idx[c] for c in self.texts[idx]
                         if c in self.char_to_idx
                         and self.char_to_idx[c] != BLANK_IDX]
        label_tensor  = torch.tensor(label_indices, dtype=torch.long)
        return image_tensor, label_tensor, len(label_indices)


def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images        = torch.stack(images, 0)
    labels        = torch.cat(labels, 0)
    label_lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels, label_lengths, None

# ============================================================================
# PART 10: TRAINING
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss, n = 0.0, 0
    bar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, labels, label_lengths, _ in bar:
        images        = images.to(device)
        labels        = labels.to(device)
        label_lengths = label_lengths.to(device)
        optimizer.zero_grad()
        with autocast(device_type=device, enabled=(device == 'cuda')):
            logits     = model(images)
            logits_ctc = logits.permute(1, 0, 2)
            log_probs  = F.log_softmax(logits_ctc, dim=-1)
            T          = logits_ctc.size(0)
            B          = images.size(0)
            input_lens = torch.full((B,), T, dtype=torch.long, device=device)
            loss       = criterion(log_probs, labels, input_lens, label_lengths)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        n          += 1
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / n if n else 0.0


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss, n       = 0.0, 0
    all_preds, all_conf = [], []
    with torch.no_grad():
        for images, labels, label_lengths, _ in tqdm(loader, desc="Evaluating"):
            images        = images.to(device)
            labels        = labels.to(device)
            label_lengths = label_lengths.to(device)
            logits     = model(images)
            logits_ctc = logits.permute(1, 0, 2)
            log_probs  = F.log_softmax(logits_ctc, dim=-1)
            T, B       = logits_ctc.size(0), images.size(0)
            input_lens = torch.full((B,), T, dtype=torch.long, device=device)
            loss       = criterion(log_probs, labels, input_lens, label_lengths)
            total_loss += loss.item()
            n          += 1
            texts, confs = model.decode_predictions(logits)
            all_conf.extend(confs)
            all_preds.extend(texts)
    return {
        'loss':        total_loss / n if n else 0.0,
        'confidence':  float(np.mean(all_conf)) if all_conf else 0.0,
        'predictions': all_preds[:10]}


def train_model(model, train_loader, test_loader, device,
                num_epochs=100, lr=1e-4,
                save_path='major_project_trained_model.keras',
                early_stop_patience=15, min_delta=1e-4):
    print("\n" + "=" * 70)
    print("STARTING TRAINING — CNN + ViT + BiLSTM Hindi Handwriting OCR")
    print(f"  Device: {device}  |  Epochs: {num_epochs}  |  LR: {lr}")
    print(f"  Train: {len(train_loader.dataset)}  |  Test: {len(test_loader.dataset)}")
    print(f"  Early stop: patience={early_stop_patience}, min_delta={min_delta}")
    print("=" * 70)

    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7)
    scaler    = GradScaler(enabled=(device == 'cuda'))

    history   = {'train_loss': [], 'test_loss': [], 'test_confidence': []}
    best_loss  = float('inf')
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*70}\nEPOCH {epoch}/{num_epochs}\n{'='*70}")
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, scaler, device, epoch)
        eval_res   = evaluate_model(model, test_loader, criterion, device)
        test_loss  = eval_res['loss']
        scheduler.step(test_loss)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_confidence'].append(eval_res['confidence'])

        print(f"\n  Train Loss: {train_loss:.4f}  |  Test Loss: {test_loss:.4f}"
              f"  |  Confidence: {eval_res['confidence']:.4f}"
              f"  |  LR: {optimizer.param_groups[0]['lr']:.2e}")
        print("  Sample predictions:", eval_res['predictions'][:5])

        if test_loss < best_loss - min_delta:
            best_loss  = test_loss
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss,
                'history': history}, save_path)
            print(f"  Saved best model (loss={test_loss:.4f})")
        else:
            no_improve += 1
            remaining   = early_stop_patience - no_improve
            print(f"  No improvement for {no_improve}/{early_stop_patience} epochs "
                  f"(best={best_loss:.4f}) — {remaining} left before early stop")
            if no_improve >= early_stop_patience:
                print(f"\nEarly stopping triggered at epoch {epoch}. "
                      f"Best loss: {best_loss:.4f}")
                break

    print(f"\nTraining done. Best test loss: {best_loss:.4f}  ->  {save_path}")
    return history

# ============================================================================
# PART 11: WEBCAM PREPROCESSOR
# ============================================================================
class WebcamPreprocessor:
    def __init__(self, target_size=(64, 256), device='cpu'):
        self.target_size = target_size
        self.device      = device

    def preprocess_frame(self, frame):
        image  = preprocess_image(frame, self.target_size)
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor

    def extract_text_region(self, frame, padding=15):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        fh, fw     = frame.shape[:2]
        min_area   = fh * fw * 0.002
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        valid = []
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < 20 or h < 8:
                continue
            valid.append((x, y, w, h))

        if valid:
            x1  = max(0,  min(r[0]        for r in valid) - padding)
            y1  = max(0,  min(r[1]        for r in valid) - padding)
            x2  = min(fw, max(r[0] + r[2] for r in valid) + padding)
            y2  = min(fh, max(r[1] + r[3] for r in valid) + padding)
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0 and (x2 - x1) > 20 and (y2 - y1) > 8:
                return roi, (x1, y1, x2, y2)
        return frame, None

# ============================================================================
# PART 12: REAL-TIME INFERENCE ENGINE
# ============================================================================
class RealtimeInferenceEngine:
    def __init__(self, model, preprocessor, confidence_threshold=0.5):
        self.model        = model
        self.preprocessor = preprocessor
        self.threshold    = confidence_threshold
        self._last_debug  = None
        self.model.eval()

    @torch.no_grad()
    def predict(self, frame):
        tensor      = self.preprocessor.preprocess_frame(frame)
        debug_img   = (tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
        debug_display = cv2.resize(debug_img, (512, 128))
        cv2.putText(debug_display, 'Model Input (dark text on white)',
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,), 1)
        cv2.namedWindow("Processed Input", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Processed Input", 512, 128)
        cv2.imshow("Processed Input", debug_display)
        self._last_debug = debug_display
        logits       = self.model(tensor)
        texts, confs = self.model.decode_predictions(logits)
        return texts[0], confs[0]

    def run_webcam_loop(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        print("Webcam running — press 'q' to quit, 's' to save frame")
        frame_count = 0
        prediction  = ""
        confidence  = 0.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            display   = frame.copy()
            roi, bbox = self.preprocessor.extract_text_region(frame)
            if bbox:
                cv2.rectangle(display, bbox[:2], bbox[2:], (0, 255, 0), 2)
            if frame_count % 3 == 0:
                try:
                    prediction, confidence = self.predict(roi if bbox else frame)
                except Exception as e:
                    prediction, confidence = f"Error: {e}", 0.0
            display = self._overlay(display, prediction, confidence)
            cv2.imshow("Hindi OCR", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                debug_panel = getattr(self, '_last_debug', None)
                if debug_panel is not None:
                    debug_bgr = cv2.cvtColor(debug_panel, cv2.COLOR_GRAY2BGR)
                    dh, dw    = display.shape[:2]
                    debug_bgr = cv2.resize(debug_bgr, (dw, dh))
                    combined  = np.hstack([display, debug_bgr])
                else:
                    combined  = display
                fname = f"capture_{frame_count}.png"
                cv2.imwrite(fname, combined)
                print(f"Saved {fname}")
            frame_count += 1
        cap.release()
        cv2.destroyAllWindows()

    def _overlay(self, frame, text, confidence):
        h, w    = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 100), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        conf_color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
        cv2.putText(frame, f"Conf: {confidence:.2%}", (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
        pred_color = (0, 255, 0) if confidence > 0.5 else (0, 255, 255)
        cv2.putText(frame, f"Pred: {text}", (20, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, pred_color, 2)
        return frame

# ============================================================================
# PART 13: MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("   CNN + ViT + BiLSTM — Hindi Handwriting OCR")
    print("   Train: sikhna.parquet  |  Test: pariksha.parquet")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[INFO] Device: {device.upper()}")

    TRAIN_PARQUET = "sikhna.parquet"
    TEST_PARQUET  = "pariksha.parquet"
    TARGET_SIZE   = (64, 256)
    MODEL_SAVE    = "LSTM_VERSION.keras"
    CHAR_LIST     = DEVANAGARI_CHARS

    cuda_available      = torch.cuda.is_available()
    BATCH_SIZE          = 32 if cuda_available else 16
    NUM_EPOCHS          = 50 if cuda_available else 20
    EARLY_STOP_PATIENCE = 15
    MIN_DELTA           = 1e-4
    LEARNING_RATE       = 1e-4
    NUM_WORKERS         = 4 if cuda_available else 0
    MAX_TRAIN           = None if cuda_available else 5000
    MAX_TEST            = None if cuda_available else 1000
    PATCH_SIZE          = 4 if cuda_available else 8

    print(f"\n[CONFIG]")
    print(f"  Batch: {BATCH_SIZE}  |  Epochs: {NUM_EPOCHS}  |  LR: {LEARNING_RATE}")
    print(f"  Image size: {TARGET_SIZE}  |  Patch size: {PATCH_SIZE}  |  Classes: {NUM_CLASSES}")
    print(f"  Early stop: patience={EARLY_STOP_PATIENCE}, min_delta={MIN_DELTA}")
    print(f"  Architecture: CNN -> ViT (depth=6) -> BiLSTM (2L, hidden=256) -> CTC")
    if not cuda_available:
        print(f"  [CPU mode] capped at {MAX_TRAIN} train / {MAX_TEST} test samples")

    print("\n" + "=" * 70)
    print("  1. Train from scratch")
    print("  2. Load model + run inference")
    print("  3. Train then run inference")
    print("=" * 70)
    try:
        mode = input("\nChoice (1/2/3): ").strip()
    except Exception:
        mode = "1"

    print("\n[STEP 1] Building model...")
    model = HybridCNNViT_HindiOCR(
        input_channels=1, feature_dim=256, embed_dim=384,
        vit_depth=6, vit_heads=6, num_classes=NUM_CLASSES,
        patch_size=PATCH_SIZE, bilstm_hidden=256, bilstm_layers=2,
        pretrained_vit=False, char_list=CHAR_LIST).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}  (~{total_params*4/1024/1024:.1f} MB)")
    print(f"  Layers: CNN backbone -> ViT (6 blocks) -> BiLSTM (2 layers) -> CTC head")

    if mode in ('1', '3'):
        print("\n[STEP 2] Loading datasets...")
        train_ds = ParquetHandwritingDataset(
            TRAIN_PARQUET, CHAR_LIST, TARGET_SIZE, augment=True,  max_samples=MAX_TRAIN)
        test_ds  = ParquetHandwritingDataset(
            TEST_PARQUET,  CHAR_LIST, TARGET_SIZE, augment=False, max_samples=MAX_TEST)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, collate_fn=collate_fn,
                                  pin_memory=(device == 'cuda'))
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, collate_fn=collate_fn,
                                  pin_memory=(device == 'cuda'))
        print("\n[STEP 3] Training...")
        train_model(model, train_loader, test_loader, device,
                    num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, save_path=MODEL_SAVE,
                    early_stop_patience=EARLY_STOP_PATIENCE, min_delta=MIN_DELTA)

    if mode in ('2', '3'):
        print("\n[STEP 4] Loading checkpoint...")
        if Path(MODEL_SAVE).exists():
            ckpt = torch.load(MODEL_SAVE, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            print(f"  Loaded epoch {ckpt['epoch']}  |  test_loss={ckpt['test_loss']:.4f}")
        else:
            print(f"  [WARNING] {MODEL_SAVE} not found — using untrained weights")

        print("\n[STEP 5] Webcam inference")
        try:
            run_cam = input("Start webcam? (y/n): ").strip().lower()
        except Exception:
            run_cam = 'n'
        if run_cam == 'y':
            preprocessor = WebcamPreprocessor(target_size=TARGET_SIZE, device=device)
            engine       = RealtimeInferenceEngine(model, preprocessor)
            try:
                engine.run_webcam_loop(camera_id=0)
            except KeyboardInterrupt:
                print("\nStopped by user")
        else:
            print("Skipping webcam.")

    print("\n" + "=" * 70)
    print("SESSION ENDED")
    print("=" * 70)


if __name__ == "__main__":
    main()
