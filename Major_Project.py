#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import timm
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

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
    'ढ़', 'फ़', 'य़', 'ॠ', 'ॢ', '।', '॥', '०', '१', '२', '३', '४', '५', '६',
    '७', '८', '९', '॰', 'ॱ', 'ॲ', 'ॻ', 'ॼ', 'ॽ', 'ॾ',
    '<BLANK>'
]

BLANK_IDX   = len(DEVANAGARI_CHARS) - 1   # 108
NUM_CLASSES = len(DEVANAGARI_CHARS)        # 109

# ============================================================================
# FIX 1 — SHARED PREPROCESSING FUNCTION (CRITICAL)
# BUG: Training used raw PIL→grayscale→resize→/255 (dark text on light bg).
#      Webcam used adaptiveThreshold BINARY_INV (WHITE text on BLACK bg).
#      This is a complete polarity inversion — model always saw the opposite.
# FIX: Single preprocess_image() used by BOTH dataset AND webcam inference.
#      Pipeline: grayscale → CLAHE contrast enhance → aspect-ratio resize
#      with white padding → normalize /255.
#      Result: dark strokes on light background, identical in both paths.
# ============================================================================
def preprocess_image(image_input, target_size=(64, 256)):
    """
    Shared preprocessing for training AND inference.
    Accepts: numpy uint8 HxW (gray), HxWx3 (BGR), or PIL Image.
    Returns: float32 numpy array (H, W) in [0,1].
             Light background (~0.8+), dark text strokes (~0.0-0.3).
    """
    H, W = target_size

    # Convert to grayscale uint8
    if isinstance(image_input, Image.Image):
        img = np.array(image_input.convert('L'), dtype=np.uint8)
    elif image_input.ndim == 3:
        img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
    else:
        img = image_input.copy()

    # CLAHE: improve local contrast (helps faint ink strokes)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # FIX 2 — ASPECT-RATIO PRESERVING RESIZE WITH WHITE PADDING
    # Bug: original code squashed images with cv2.resize(img, (W, H))
    # which distorts character proportions, especially for tall Devanagari.
    ih, iw = img.shape
    scale  = min(W / iw, H / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((H, W), 255, dtype=np.uint8)   # white background
    y_off  = (H - nh) // 2
    x_off  = (W - nw) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized

    return canvas.astype(np.float32) / 255.0

# ============================================================================
# PART 1: CNN FEATURE EXTRACTOR (unchanged)
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
# PART 2: TOKEN EMBEDDING (unchanged)
# ============================================================================
class TokenEmbedding(nn.Module):
    def __init__(self, feature_dim=256, embed_dim=384, patch_size=4):
        super().__init__()
        self.projection = nn.Conv2d(feature_dim, embed_dim,
                                    kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # (B, N, D)
        return x, (H, W)


# ============================================================================
# PART 3: POSITIONAL ENCODING (unchanged)
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len=1000, embed_dim=384):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02)

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]


# ============================================================================
# PART 4: VISION TRANSFORMER ENCODER (unchanged)
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
# PART 5: CTC PREDICTION HEAD (unchanged)
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
# PART 6: COMPLETE HYBRID MODEL
#
# FIX 3 — CTC DECODING BUG:
#   Bug: original `prev` was only updated when a non-blank char was emitted.
#        So "क blank blank क" → prev never became BLANK_IDX between the two
#        क's → second क was suppressed (treated as repeat) → missing chars.
#   Fix: `prev` is updated on EVERY token (including blanks), which is the
#        correct CTC collapse rule: collapse consecutive identical tokens,
#        then remove blanks.
#
# FIX 4 — CONFIDENCE CALCULATION:
#   Bug: original used mean(max_prob over ALL timesteps including blank-
#        dominated ones). With random weights, blank gets ~1/109 ≈ 0.9%
#        which is why confidence showed ~0.00%.
#   Fix: confidence = exp(mean log-prob of greedy path over NON-BLANK
#        emitted characters only). This gives a true per-character
#        probability that reflects how certain the model is about each
#        character it actually predicted.
# ============================================================================
class HybridCNNViT_HindiOCR(nn.Module):
    def __init__(self, input_channels=1, feature_dim=256, embed_dim=384,
                 vit_depth=6, vit_heads=6, num_classes=NUM_CLASSES,
                 patch_size=4, pretrained_vit=False, char_list=None):
        super().__init__()
        self.cnn_backbone    = CNNFeatureExtractor(input_channels, feature_dim)
        self.token_embedding = TokenEmbedding(feature_dim, embed_dim, patch_size)
        self.pos_encoding    = PositionalEncoding(1000, embed_dim)
        self.vit_encoder     = ViTEncoder(embed_dim, vit_depth, vit_heads, pretrained_vit)
        self.prediction_head = CTCPredictionHead(embed_dim, num_classes)
        self.char_list = char_list if char_list is not None else DEVANAGARI_CHARS

    def forward(self, x, return_features=False):
        cnn_feat = self.cnn_backbone(x)
        tokens, spatial = self.token_embedding(cnn_feat)
        tokens   = self.pos_encoding(tokens)
        vit_feat = self.vit_encoder(tokens)
        logits   = self.prediction_head(vit_feat)
        if return_features:
            return logits, {'cnn_features': cnn_feat, 'tokens': tokens,
                            'vit_features': vit_feat, 'spatial_shape': spatial}
        return logits

    def decode_predictions(self, logits):
        """
        Greedy CTC decode — returns (list[str], list[float]).
        FIX 3: prev tracks ALL tokens so repeated chars separated by blanks
                are correctly preserved (e.g. "कक" stays "कक" not "क").
        FIX 4: confidence = exp(mean log-prob of emitted non-blank chars).
        """
        log_probs = F.log_softmax(logits, dim=-1)   # (B, T, C)
        preds     = torch.argmax(log_probs, dim=-1)  # (B, T)

        decoded_texts = []
        decoded_confs = []

        for b in range(preds.size(0)):
            seq    = preds[b].cpu().numpy()
            lp_seq = log_probs[b].cpu().numpy()   # (T, C)
            chars, lp_chars = [], []
            prev = None   # FIX 3: track ALL tokens, not just non-blank

            for t, idx in enumerate(seq):
                if idx != BLANK_IDX and idx != prev:
                    chars.append(self.char_list[idx])
                    lp_chars.append(lp_seq[t, idx])
                prev = idx   # FIX 3: always update prev

            decoded_texts.append(''.join(chars))
            # FIX 4: mean prob over emitted chars only
            decoded_confs.append(float(np.exp(np.mean(lp_chars))) if lp_chars else 0.0)

        return decoded_texts, decoded_confs

# ============================================================================
# PART 7: PARQUET DATASET
# FIX 1 applied: uses shared preprocess_image() instead of raw squash-resize.
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
        print(f"  → {len(self.texts)} usable samples")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        raw       = self.image_bytes[idx]
        img_bytes = raw['bytes'] if isinstance(raw, dict) else raw
        pil_img   = Image.open(BytesIO(img_bytes)).convert('L')

        if self.augment:
            img_np  = np.array(pil_img, dtype=np.uint8)
            img_np  = self._augment(img_np)
            pil_img = Image.fromarray(img_np)

        # FIX 1: shared preprocessing — identical to webcam path
        image        = preprocess_image(pil_img, self.target_size)
        image_tensor = torch.from_numpy(image).unsqueeze(0)   # (1, H, W)

        label_indices = [self.char_to_idx[c] for c in self.texts[idx]
                         if c in self.char_to_idx
                         and self.char_to_idx[c] != BLANK_IDX]
        label_tensor  = torch.tensor(label_indices, dtype=torch.long)
        return image_tensor, label_tensor, len(label_indices)

    def _augment(self, image):
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-8, 8)
            h, w  = image.shape
            M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h),
                                   borderMode=cv2.BORDER_REPLICATE)
        if np.random.rand() < 0.5:
            image = np.clip(image * np.random.uniform(0.8, 1.2),
                            0, 255).astype(np.uint8)
        if np.random.rand() < 0.3:
            image = np.clip(image + np.random.normal(0, 5, image.shape),
                            0, 255).astype(np.uint8)
        return image


def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images        = torch.stack(images, 0)
    labels        = torch.cat(labels, 0)
    label_lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels, label_lengths, None

# ============================================================================
# PART 8: TRAINING
# FIX 5 — EVALUATE CONFIDENCE: original evaluate_model computed confidence
#   as mean(max softmax prob over ALL timesteps), which is dominated by blank
#   tokens and gives near-zero values. Now delegates to decode_predictions()
#   which uses the corrected per-character log-prob confidence (FIX 4).
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
            logits     = model(images)                      # (B, T, C)
            logits_ctc = logits.permute(1, 0, 2)           # (T, B, C)
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
        n += 1
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / n if n else 0.0


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
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
            n += 1
            # FIX 5: use decode_predictions for correct confidence
            texts, confs = model.decode_predictions(logits)
            all_conf.extend(confs)
            all_preds.extend(texts)
    return {
        'loss':        total_loss / n if n else 0.0,
        'confidence':  float(np.mean(all_conf)) if all_conf else 0.0,
        'predictions': all_preds[:10]
    }


def train_model(model, train_loader, test_loader, device,
                num_epochs=50, lr=1e-4, save_path='hindi_ocr_cnn_vit.pth'):
    print("\n" + "=" * 70)
    print("STARTING TRAINING — Hindi Handwriting OCR (Parquet)")
    print(f"  Device: {device}  |  Epochs: {num_epochs}  |  LR: {lr}")
    print(f"  Train: {len(train_loader.dataset)}  |  Test: {len(test_loader.dataset)}")
    print("=" * 70)

    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    scaler    = GradScaler(enabled=(device == 'cuda'))
    history   = {'train_loss': [], 'test_loss': [], 'test_confidence': []}
    best_loss = float('inf')

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

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'test_loss': test_loss,
                        'history': history}, save_path)
            print(f"  Saved best model (loss={test_loss:.4f})")

    print(f"\nTraining done. Best test loss: {best_loss:.4f}  →  {save_path}")
    return history

# ============================================================================
# UNICODE TEXT RENDERER (Pillow-based — required for Devanagari)
# WHY: cv2.putText only covers ASCII (0-127). Devanagari needs glyph shaping
#      (matras, half-forms, conjuncts) via FreeType + TTF. OpenCV renders
#      every Devanagari codepoint as a box or skips it silently.
# ============================================================================
_FONT_CACHE = {}

def _ensure_devanagari_font():
    candidates = [
        'NotoSansDevanagari-Regular.ttf',
        'C:/Windows/Fonts/mangal.ttf',
        'C:/Windows/Fonts/Mangal.ttf',
        'C:/Windows/Fonts/NotoSansDevanagari-Regular.ttf',
        '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf',
        '/usr/share/fonts/noto/NotoSansDevanagari-Regular.ttf',
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    import urllib.request
    url  = ('https://github.com/googlefonts/noto-fonts/raw/main/'
            'hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf')
    dest = 'NotoSansDevanagari-Regular.ttf'
    print('[Font] Downloading NotoSansDevanagari-Regular.ttf ...')
    try:
        urllib.request.urlretrieve(url, dest)
        return dest
    except Exception as e:
        print(f'[Font] Download failed: {e}')
        return None


def _load_devanagari_font(size=28):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    path = _ensure_devanagari_font()
    if path and Path(path).exists():
        font = ImageFont.truetype(path, size)
        _FONT_CACHE[size] = font
        return font
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _draw_unicode_text(frame_bgr, text, pos, font_size=28, color=(0, 255, 0)):
    """Render Unicode (Devanagari) text onto an OpenCV BGR frame via Pillow."""
    font      = _load_devanagari_font(font_size)
    pil_img   = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw      = ImageDraw.Draw(pil_img)
    rgb_color = (color[2], color[1], color[0])   # BGR → RGB
    draw.text(pos, text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ============================================================================
# PART 9: WEBCAM PREPROCESSOR
#
# FIX 1 applied: preprocess_frame() calls shared preprocess_image().
#   BEFORE: BGR→gray → adaptiveThreshold BINARY_INV → resize → /255
#           = white text on black background (INVERTED vs training)
#   AFTER:  BGR→gray → CLAHE → aspect-ratio resize + white pad → /255
#           = dark text on light background (IDENTICAL to training)
#
# FIX 6 — ROI EXTRACTION:
#   Bug: area > 100 threshold was too small — webcam noise blobs (dust,
#        shadows) were included, expanding the bounding box to the whole
#        frame and feeding garbage to the model.
#   Fix: Otsu threshold + morphological close to merge strokes into word
#        blobs, then filter by relative area (≥0.2% of frame) and minimum
#        width/height. Also draws a guide rectangle on the live feed so the
#        user knows where to hold the paper.
# ============================================================================
class WebcamPreprocessor:
    def __init__(self, target_size=(64, 256), device='cpu'):
        self.target_size = target_size
        self.device      = device

    def preprocess_frame(self, frame):
        """Convert webcam BGR frame → model tensor (1, 1, H, W)."""
        image  = preprocess_image(frame, self.target_size)   # FIX 1
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor

    def extract_text_region(self, frame, padding=15):
        """
        FIX 6: robust ROI extraction using Otsu + morphological close.
        Returns (roi_bgr, (x1,y1,x2,y2)) or (frame, None).
        """
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        # Otsu: more stable than adaptive for ROI detection
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological close: join nearby strokes into word-level blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        fh, fw   = frame.shape[:2]
        min_area = (fw * fh) * 0.002   # ≥ 0.2% of frame

        valid = []
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < 20 or h < 8:   # reject thin noise blobs
                continue
            valid.append((x, y, w, h))

        if valid:
            x1 = max(0,  min(r[0]        for r in valid) - padding)
            y1 = max(0,  min(r[1]        for r in valid) - padding)
            x2 = min(fw, max(r[0] + r[2] for r in valid) + padding)
            y2 = min(fh, max(r[1] + r[3] for r in valid) + padding)
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0 and (x2 - x1) > 20 and (y2 - y1) > 8:
                return roi, (x1, y1, x2, y2)

        return frame, None

# ============================================================================
# PART 10: REAL-TIME INFERENCE ENGINE
#
# FIX 3 + FIX 4 applied via decode_predictions().
# FIX 7 — DEBUG WINDOW: original showed the raw tensor which could be
#   all-black if preprocessing was wrong. Now shows the actual preprocessed
#   image that goes into the model, with a label, so you can visually verify
#   it matches training data (dark strokes on light background).
# FIX 8 — model.eval() + torch.no_grad(): original had model.eval() in
#   __init__ but predict() lacked @torch.no_grad(), so BatchNorm and Dropout
#   were still in eval mode but gradients were being tracked, wasting memory
#   and occasionally causing incorrect BN statistics. Now both are enforced.
# ============================================================================
class RealtimeInferenceEngine:
    def __init__(self, model, preprocessor, confidence_threshold=0.5):
        self.model      = model
        self.preprocessor = preprocessor
        self.threshold  = confidence_threshold
        self._last_debug = None
        self.model.eval()   # FIX 8: ensure eval mode

    @torch.no_grad()   # FIX 8: no gradient tracking during inference
    def predict(self, frame):
        tensor = self.preprocessor.preprocess_frame(frame)

        # FIX 7: debug window shows the actual model input
        debug_img = (tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
        debug_display = cv2.resize(debug_img, (512, 128))
        # Add label so it's clear what we're looking at
        cv2.putText(debug_display, 'Model Input (should be dark text on white)',
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,), 1)
        cv2.namedWindow("Processed Input", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Processed Input", 512, 128)
        cv2.imshow("Processed Input", debug_display)
        self._last_debug = debug_display

        logits = self.model(tensor)   # (1, T, C)

        # FIX 3 + FIX 4: decode_predictions returns (texts, confs)
        texts, confs = self.model.decode_predictions(logits)
        return texts[0], confs[0]

    def run_webcam_loop(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        print("Webcam running — press 'q' to quit, 's' to save frame")

        frame_count  = 0
        prediction   = ""
        confidence   = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
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
                    cv2.putText(debug_bgr, 'Processed Input', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    combined = np.hstack([display, debug_bgr])
                else:
                    combined = display
                fname = f"capture_{frame_count}.png"
                cv2.imwrite(fname, combined)
                print(f"Saved {fname}")

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

    def _overlay(self, frame, text, confidence):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 100), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        conf_color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
        cv2.putText(frame, f"Conf: {confidence:.2%}", (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)

        # Devanagari MUST use Pillow — cv2.putText cannot render it
        pred_color = (0, 255, 0) if confidence > 0.5 else (0, 255, 255)
        frame = _draw_unicode_text(frame, f"Pred: {text}",
                                   pos=(20, h - 70),
                                   font_size=30,
                                   color=pred_color)
        return frame

# ============================================================================
# PART 11: MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("   HYBRID CNN + ViT — Hindi Handwriting OCR  [FIXED]")
    print("   Train: sikhna.parquet  |  Test: pariksha.parquet")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[INFO] Device: {device.upper()}")

    TRAIN_PARQUET = "sikhna.parquet"
    TEST_PARQUET  = "pariksha.parquet"
    TARGET_SIZE   = (64, 256)
    MODEL_SAVE    = "major_project_trained_model.keras"
    CHAR_LIST     = DEVANAGARI_CHARS

    cuda_available = torch.cuda.is_available()
    BATCH_SIZE     = 16
    NUM_EPOCHS     = 50 if cuda_available else 5
    LEARNING_RATE  = 1e-4
    NUM_WORKERS    = 4 if cuda_available else 0
    MAX_TRAIN      = None if cuda_available else 5000
    MAX_TEST       = None if cuda_available else 1000

    print(f"\n[CONFIG]")
    print(f"  Batch: {BATCH_SIZE}  |  Epochs: {NUM_EPOCHS}  |  LR: {LEARNING_RATE}")
    print(f"  Image size: {TARGET_SIZE}  |  Classes: {NUM_CLASSES}")
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
        patch_size=4, pretrained_vit=True, char_list=CHAR_LIST
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}  (~{total_params*4/1024/1024:.1f} MB)")

    if mode in ('1', '3'):
        print("\n[STEP 2] Loading datasets...")
        train_ds = ParquetHandwritingDataset(TRAIN_PARQUET, CHAR_LIST, TARGET_SIZE,
                                             augment=True,  max_samples=MAX_TRAIN)
        test_ds  = ParquetHandwritingDataset(TEST_PARQUET,  CHAR_LIST, TARGET_SIZE,
                                             augment=False, max_samples=MAX_TEST)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, collate_fn=collate_fn,
                                  pin_memory=(device == 'cuda'))
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, collate_fn=collate_fn,
                                  pin_memory=(device == 'cuda'))
        print("\n[STEP 3] Training...")
        train_model(model, train_loader, test_loader, device,
                    num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, save_path=MODEL_SAVE)

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


# In[ ]:




