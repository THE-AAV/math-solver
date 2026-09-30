"""Step 2: run the demo.   Run:  python app.py   then open the link it prints."""
import cv2
import gradio as gr
import numpy as np
import tensorflow as tf

from utils import CLASSES, MODEL_PATH, normalize

model = tf.keras.models.load_model(MODEL_PATH)

TO_PYTHON = {"times": "*", "div": "/"}   # what we evaluate
TO_SHOW = {"times": "×", "div": "÷"}      # what we display


def to_gray(image):
    """Convert sketchpad / uploaded image to a grayscale numpy array."""
    if isinstance(image, dict):              # Sketchpad returns a dict
        image = image.get("composite")
    if image is None:
        return None
    img = np.array(image)
    if img.ndim == 3 and img.shape[2] == 4:  # transparent background -> white
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape
    if w > 1200:                             # shrink large phone photos
        img = cv2.resize(img, (1200, int(h * 1200 / w)))
    return img


def find_symbols(gray):
    """Split the equation into single symbols, returned left to right."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[2] * b[3] > 15]      # drop tiny specks
    boxes.sort(key=lambda b: b[0])

    # merge pieces stacked vertically (e.g. the two dots and line of ÷)
    merged = []
    for x, y, w, h in boxes:
        if merged:
            mx, my, mw, mh = merged[-1]
            overlap = min(mx + mw, x + w) - max(mx, x)
            if overlap > 0.5 * min(mw, w):
                nx, ny = min(mx, x), min(my, y)
                merged[-1] = (nx, ny, max(mx + mw, x + w) - nx, max(my + mh, y + h) - ny)
                continue
        merged.append((x, y, w, h))
    return binary, merged


def solve(image):
    gray = to_gray(image)
    if gray is None:
        return None, "", "Draw or upload an equation first."
    binary, boxes = find_symbols(gray)
    if not boxes:
        return None, "", "No symbols found."

    batch = np.array([normalize(binary[y:y + h, x:x + w]) for x, y, w, h in boxes])
    probs = model.predict(batch[..., np.newaxis], verbose=0)
    labels = [CLASSES[i] for i in probs.argmax(axis=1)]

    expr_python = "".join(TO_PYTHON.get(s, s) for s in labels)
    expr_show = " ".join(TO_SHOW.get(s, s) for s in labels)

    try:
        result = eval(expr_python, {"__builtins__": {}}, {})  # only digits/operators possible
        if isinstance(result, float):
            result = int(result) if result.is_integer() else round(result, 4)
        answer = str(result)
    except ZeroDivisionError:
        answer = "Cannot divide by zero"
    except Exception:
        answer = "Could not solve (check the recognised expression)"

    # draw boxes + predictions so the audience can see what the CNN saw
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for (x, y, w, h), s, p in zip(boxes, labels, probs.max(axis=1)):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 160, 0), 2)
        cv2.putText(vis, f"{TO_PYTHON.get(s, s)} {p:.0%}", (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 0, 0), 2)
    return vis, expr_show, answer


with gr.Blocks(title="Handwritten Math Solver") as demo:
    gr.Markdown("# ✍️ Handwritten Math Equation Solver\n"
                "Write an equation using digits, **+ − × ÷** and brackets, then press **Solve**.")
    with gr.Row():
        with gr.Column():
            with gr.Tab("Draw"):
                sketch = gr.Sketchpad(type="numpy", label="Draw here",
                                      brush=gr.Brush(default_size=6, colors=["#000000"],
                                                     color_mode="fixed"))
                draw_btn = gr.Button("Solve", variant="primary")
            with gr.Tab("Upload photo"):
                upload = gr.Image(type="numpy", label="Photo of equation (dark pen, white paper)")
                upload_btn = gr.Button("Solve", variant="primary")
        with gr.Column():
            seen = gr.Image(label="What the model detected")
            expr = gr.Textbox(label="Recognised expression")
            answer = gr.Textbox(label="Answer")

    outputs = [seen, expr, answer]
    draw_btn.click(solve, sketch, outputs)
    upload_btn.click(solve, upload, outputs)

if __name__ == "__main__":
    demo.launch()
