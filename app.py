import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import gradio as gr
import tempfile
import os

# ----------------- GLOBAL VARIABLES -------------------
X, y = None, None
X_train, X_test, y_train, y_test = None, None, None, None


# ---------- BUTTON 1: GENERATE DATA + TRAIN/TEST SPLIT ----------
def split_dataset(test_ratio):

    global X, y, X_train, X_test, y_train, y_test

    X, y = datasets.make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=2.0,
        random_state=None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=None
    )

    return f"Dataset split successfully!\nTrain size: {len(X_train)}\nTest size: {len(X_test)}"


# ---------- BUTTON 2: VISUALIZE KNN + ACCURACY ----------
def visualize_knn(n_neighbors):

    global X_train, X_test, y_train, y_test

    if X_train is None:
        return None, "⚠ Please click 'Split Dataset' first!"

    n_neighbors = int(n_neighbors)

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    x_min, x_max = min(X_train[:, 0].min(), X_test[:, 0].min()) - 1, max(X_train[:, 0].max(), X_test[:, 0].max()) + 1
    y_min, y_max = min(X_train[:, 1].min(), X_test[:, 1].min()) - 1, max(X_train[:, 1].max(), X_test[:, 1].max()) + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 7))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap="Accent")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="Accent", edgecolors="black", marker="o", label="Train")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="Accent", edgecolors="black", marker="^", label="Test")
    plt.legend()
    plt.title(f"KNN Decision Boundary (k = {n_neighbors})")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp_file.name)
    plt.close()

    return temp_file.name, f"Accuracy: {acc:.4f}"


# -------------------- CUSTOM CSS --------------------
custom_css = """
.gr-button {
    background-color: #007bff !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 12px 20px !important;
    font-weight: bold !important;
}
.gr-slider input {
    accent-color: #007bff !important;
}
body, .gradio-container {
    background: #1f1f1f !important;
    color: white !important;
}
.gr-box, .gr-textbox, .gr-markdown {
    color: white !important;
}
"""


# -------------------- GRADIO UI --------------------
with gr.Blocks(css=custom_css) as demo:

    gr.Markdown("## 🧠 KNN Decision Boundary + Dynamic Train/Test Split + Enhanced UI")

    with gr.Row():  # ---- 2 column layout ----
        with gr.Column(scale=1):  # LEFT SIDE — Parameters
            split_ratio = gr.Slider(0.1, 0.5, value=0.3, step=0.05, label="Test Size Ratio")
            split_btn = gr.Button("Split Dataset")
            split_output = gr.Textbox(label="Split Result", interactive=False)

            k_slider = gr.Slider(1, 20, value=3, step=1, label="K Value (n_neighbors)")
            visualize_btn = gr.Button("Visualize")

        with gr.Column(scale=2):  # RIGHT SIDE — Visualization
            output_img = gr.Image()
            accuracy_text = gr.Textbox(label="Model Accuracy", interactive=False)

    split_btn.click(
        split_dataset,
        inputs=[split_ratio],
        outputs=[split_output]
    )

    visualize_btn.click(
        visualize_knn,
        inputs=[k_slider],
        outputs=[output_img, accuracy_text]
    )


demo.launch(debug=True)
