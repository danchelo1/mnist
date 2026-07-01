import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

m1_path = "models/model_mnist.h5"
m2_path = "models/model_augmented.h5"

if not (os.path.exists(m1_path) and os.path.exists(m2_path)):
    raise FileNotFoundError("Не найдены модели.")

model1 = keras.models.load_model(m1_path)
model2 = keras.models.load_model(m2_path)

xA_test = np.load("models/meta_test_A.npy")
yA_test = np.load("models/meta_test_yA.npy")
xB_test = np.load("models/meta_test_B.npy")
yB_test = np.load("models/meta_test_yB.npy")

y1_pred = np.argmax(model1.predict(xA_test), axis=1)
y2_pred = np.argmax(model2.predict(xB_test), axis=1)

print("Model 1 (trained on MNIST) — evaluation on its test set:")
print("Accuracy:", accuracy_score(yA_test, y1_pred))
print("Precision (macro):", precision_score(yA_test, y1_pred, average='macro'))
print("Recall (macro):", recall_score(yA_test, y1_pred, average='macro'))

print("Model 2 (trained on augmented) — evaluation on its test set:")
print("Accuracy:", accuracy_score(yB_test, y2_pred))
print("Precision (macro):", precision_score(yB_test, y2_pred, average='macro'))
print("Recall (macro):", recall_score(yB_test, y2_pred, average='macro'))

cm1 = confusion_matrix(yA_test, y1_pred)
cm2 = confusion_matrix(yB_test, y2_pred)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion matrix — Model 1 (MNIST test)")
plt.xlabel("predicted")
plt.ylabel("true")

plt.subplot(1, 2, 2)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion matrix — Model 2 (augmented test)")
plt.xlabel("predicted")
plt.ylabel("true")

plt.tight_layout()
plt.show()
