# AI-Based Crop Disease Classification Using SVM — Live Demo

## 1. What this project does

This project demonstrates an **SVM (Support Vector Machine)** classifier for
apple leaf disease classification.

The demo uses four classes:

- `healthy`
- `scab`
- `black_rot`
- `cedar_apple_rust`

The model uses:

**Image → Resize/Grayscale → HOG features → StandardScaler → RBF SVM → Prediction**

## 2. Dataset

Use the PlantVillage apple-leaf subset and arrange it like this:

```text
data/
└── apple/
    ├── healthy/
    ├── scab/
    ├── black_rot/
    └── cedar_apple_rust/
```

Put the corresponding JPG/PNG images into each folder.

## 3. Install in VS Code

Open the project folder in VS Code and run:

### Create virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

### Install packages

```powershell
pip install -r requirements.txt
```

## 4. Train the model

For a classroom demo, start with 1000 images per class:

```powershell
python train_svm.py --data data/apple --max-per-class 1000
```

This creates:

```text
outputs/
├── svm_crop_disease_model.joblib
├── results.json
└── confusion_matrix.png
```

The program prints the **actual accuracy** and classification report. Do not
invent an accuracy value; use the value printed by your run.

## 5. Start the live demo

After training:

```powershell
streamlit run app.py
```

A browser page will open.

Upload an apple leaf image and the application will show:

- the input image
- predicted class
- confidence scores
- a short explanation of how the SVM made the prediction

## 6. If the browser does not open

Run:

```powershell
python -m streamlit run app.py
```

Then open the local address displayed in the terminal.

## 7. What to say during the demo

### Dataset

> "I am using labeled apple leaf images divided into four classes: healthy,
> apple scab, black rot, and cedar apple rust."

### Preprocessing

> "The image is resized to 128 by 128 pixels, converted to grayscale, and HOG
> extracts numerical features representing the leaf's visual structure."

### SVM

> "The SVM learns decision boundaries between the four classes. I use an RBF
> kernel because the classes may not be separated by a simple straight line."

### Prediction

> "When I upload a new image, it goes through exactly the same preprocessing
> and feature extraction. The trained SVM then predicts the class."

### Results

> "The accuracy and confusion matrix shown here come from my actual test set."

## 8. Important explanation for questions

### Why not give the raw image directly to SVM?

SVM works with numerical feature vectors. HOG converts the image into a
numerical representation that the SVM can learn from.

### What is the training/test split?

The project uses an 80/20 stratified split:

- 80% training
- 20% testing

### What is the RBF kernel?

RBF allows the SVM to learn non-linear decision boundaries between classes.

### What happens to a new image?

```text
New leaf image
      ↓
Resize
      ↓
Grayscale
      ↓
HOG feature extraction
      ↓
StandardScaler
      ↓
Trained SVM
      ↓
Predicted class
```

## 9. Limitations

PlantVillage images are commonly collected under controlled conditions.
Performance on real farm photographs can therefore be different. This demo is
for classification and education; it is not a replacement for professional
plant-disease diagnosis.


