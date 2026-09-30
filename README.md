# Handwritten Math Equation Solver

Draw (or photograph) an equation like `12 + 3 × (4 − 1) ÷ 2` and the app reads it and gives the answer.

## Setup (one time)

1. Install Python 3.10, 3.11 or 3.12 (TensorFlow does not support newer versions yet).
2. In this folder, run:
   ```
   pip install -r requirements.txt
   ```
3. Download the dataset from Kaggle: **"Handwritten math symbols dataset"** by xainano
   https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols
   Unzip it and put the `extracted_images` folder inside this project folder.
   (It has many folders; the code only uses `0`–`9`, `+`, `-`, `times`, `div`, `(`, `)`.)

Your folder should look like:
```
math-solver/
├── extracted_images/   (from Kaggle)
├── utils.py
├── train.py
├── app.py
└── requirements.txt
```

## Run

```
python train.py     # trains the CNN (~5–10 min on a laptop), saves math_model.keras
python app.py       # opens the demo in your browser
```

`train.py` also prints test accuracy, per-symbol accuracy and saves `training_curves.png`. Use these in your report/slides.

## How it works (for the presentation)

1. **Input**: user draws on a canvas or uploads a photo.
2. **Segmentation (OpenCV)**: image is converted to black & white, and each connected blob of ink is found with contours. Blobs are sorted left to right. Pieces stacked on top of each other (like the dots in ÷) are merged into one symbol.
3. **Preprocessing**: each symbol is cropped, centred, padded to a square and resized to 45×45 pixels, the same way the training images are. Strokes are then thinned to a 1-pixel skeleton and redrawn at a fixed width, so a thick brush or pen looks the same to the model as the thin strokes in the dataset.
4. **Recognition (CNN)**: a 3-block convolutional neural network classifies each symbol into one of 16 classes (0–9, +, −, ×, ÷, ( and )).
5. **Solving**: the recognised symbols are joined into an expression and evaluated, respecting BODMAS.

**Model**: Data augmentation (rotation, zoom, shift) → Conv(32) → Pool → Conv(64) → Pool → Conv(128) → Pool → Dense(128) → Dropout(0.4) → Softmax(16).
Trained with Adam optimizer, sparse categorical cross-entropy loss, early stopping. 3,000 images per class max to keep training fast.

## Demo tips

- Write symbols with clear gaps between them, and don't let them touch.
- Use the multiplication sign ×, not a dot.
- For photos: dark pen on plain white paper, good lighting.
- The "What the model detected" image shows each box and its confidence, which is a nice thing to point at while presenting.
