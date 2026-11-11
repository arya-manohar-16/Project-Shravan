"""
Exports your trained keypoint classifier to model_weights.json for the web app.

Run from the project root (the folder that contains `model/`), inside the same
venv you used for keypoint_classification_EN.ipynb:

    python export_weights.py

Re-run it every time you retrain, then upload the new model_weights.json.
"""
import json
import numpy as np
import tensorflow as tf

MODEL_PATH = "model/keypoint_classifier/keypoint_classifier.hdf5"
OUT_PATH = "docs/model_weights.json"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
model.summary()

layers = []
for layer in model.layers:
    weights = layer.get_weights()
    if len(weights) == 2:                         # Dense layer: kernel + bias (Dropout has none)
        kernel, bias = weights
        act = layer.get_config().get("activation", "linear")
        layers.append({"W": kernel.tolist(), "b": bias.tolist(), "act": act})

with open(OUT_PATH, "w") as f:
    json.dump({"layers": layers}, f)


# Parity check: pure numpy forward pass (what the browser does) vs Keras
def forward(x):
    a = x
    for L in layers:
        a = a @ np.array(L["W"]) + np.array(L["b"])
        if L["act"] == "relu":
            a = np.maximum(a, 0)
        elif L["act"] == "softmax":
            e = np.exp(a - a.max())
            a = e / e.sum()
    return a


x = np.random.uniform(-1, 1, (1, 42)).astype("float32")
print("Keras :", model.predict(x, verbose=0)[0].round(4))
print("Export:", forward(x)[0].round(4))
print("Layers:", [(np.array(L["W"]).shape, L["act"]) for L in layers])
print("Wrote", OUT_PATH, "- the two printed rows above should match.")