import argparse
import json
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


IMG_SIZE = (128, 128)


def extract_features(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, IMG_SIZE)
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return features


def load_images(data_dir, max_per_class=1000):
    data_dir = Path(data_dir)
    classes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    if not classes:
        raise FileNotFoundError(
            f"No class folders found in {data_dir}. "
            "Expected folders such as healthy, scab, black_rot, cedar_apple_rust."
        )

    X, y = [], []

    for label in classes:
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            files.extend((data_dir / label).glob(ext))

        files = sorted(files)[:max_per_class]
        print(f"{label}: {len(files)} images")

        for path in files:
            image = cv2.imread(str(path))
            if image is None:
                continue
            X.append(extract_features(image))
            y.append(label)

    if not X:
        raise RuntimeError("No readable images were found.")

    return np.asarray(X), np.asarray(y), classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/apple")
    parser.add_argument("--max-per-class", type=int, default=1000)
    args = parser.parse_args()

    Path("outputs").mkdir(exist_ok=True)

    X, y, classes = load_images(args.data, args.max_per_class)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
        )),
    ])

    print("\nTraining SVM...")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    print(f"\nAccuracy: {accuracy:.4f}\n")
    print(classification_report(y_test, pred))

    report = classification_report(y_test, pred, output_dict=True)
    results = {
        "accuracy": float(accuracy),
        "classes": classes,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "classification_report": report,
        "image_size": list(IMG_SIZE),
        "feature": "HOG",
        "svm_kernel": "RBF",
        "C": 10,
    }

    with open("outputs/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    joblib.dump(model, "outputs/svm_crop_disease_model.joblib")

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, pred, labels=classes, xticks_rotation=35
    )
    plt.title("SVM Crop Disease Classification - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=180)
    plt.close()

    print("\nSaved:")
    print("  outputs/svm_crop_disease_model.joblib")
    print("  outputs/results.json")
    print("  outputs/confusion_matrix.png")
    print("\nYou can now run:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
