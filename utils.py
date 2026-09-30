"""Shared settings and preprocessing used by both train.py and app.py."""
import cv2
import numpy as np

# Symbols the model recognises (these are also the folder names in the dataset)
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           '+', '-', 'times', 'div', '(', ')']
IMG_SIZE = 45
MODEL_PATH = "math_model.keras"


def normalize(symbol):
    """Take a white-on-black image of ONE symbol and turn it into a
    45x45 float image: tightly cropped, centred, padded to a square."""
    ys, xs = np.nonzero(symbol)
    if len(xs) == 0:
        return np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
    crop = symbol[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    side = max(h, w)
    pad = max(2, int(side * 0.15))
    size = side + 2 * pad
    canvas = np.zeros((size, size), np.uint8)
    oy, ox = (size - h) // 2, (size - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop
    img = cv2.resize(canvas, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    # same stroke width for everything: the dataset has thin 1px strokes while a
    # canvas brush or pen photo gives thick ones, so thin to a skeleton and redraw
    skeleton = thin(img > 50)
    img = cv2.dilate(skeleton.astype(np.uint8) * 255, np.ones((2, 2), np.uint8))
    return img.astype(np.float32) / 255.0


def thin(mask):
    """Zhang-Suen thinning: shrink white strokes down to 1px wide lines."""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in range(2):
            p = np.pad(img, 1)
            # the 8 neighbours P2..P9, clockwise starting from the top
            n = [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:],
                 p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]
            count = sum(n)
            transitions = sum((n[i] == 0) & (n[(i + 1) % 8] == 1) for i in range(8))
            if step == 0:
                cond = (n[0] * n[2] * n[4] == 0) & (n[2] * n[4] * n[6] == 0)
            else:
                cond = (n[0] * n[2] * n[6] == 0) & (n[0] * n[4] * n[6] == 0)
            remove = (img == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & cond
            if remove.any():
                img[remove] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1].astype(bool)
