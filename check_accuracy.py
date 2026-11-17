"""
Check a saved CNN on the held-out test images – no retraining.

    python check_accuracy.py eye
    python check_accuracy.py yawn

Uses the same split (seed 42) as train_model.py, so the 218 test images were never seen in training.
Our last run: eye 97.7% (ROC-AUC 0.999), yawn 79.4% (ROC-AUC 0.889).
"""
import sys

from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

import train_model as tm

task = sys.argv[1] if len(sys.argv) > 1 else "eye"          # eye or yawn
model = tm.keras.models.load_model(f"models/{task}_model.keras")
paths, labels = tm.collect_images(tm.get_dataset_root(None), task)
_, _, (p_te, y_te) = tm.split(paths, labels)                 # same test images as training
prob = model.predict(tm.make_ds(p_te, y_te, model.input_shape[1], 64, shuffle=False), verbose=0).ravel()
pred = (prob >= 0.5).astype(int)

print(f"Test images : {len(y_te)}")
print(f"Accuracy    : {accuracy_score(y_te, pred):.3f}")
print(f"Precision   : {precision_score(y_te, pred):.3f}")
print(f"Recall      : {recall_score(y_te, pred):.3f}")
print(f"F1-score    : {f1_score(y_te, pred):.3f}")
print(f"ROC-AUC     : {roc_auc_score(y_te, prob):.3f}")
print("Confusion matrix:\n", confusion_matrix(y_te, pred))
print(classification_report(y_te, pred, target_names=tm.CLASS_NAMES[task]))
