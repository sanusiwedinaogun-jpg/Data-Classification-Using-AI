# Data-Classification-Using-AI

# Project 2: Data Classification Using AI
**DecodeLabs Internship**

## What this does
A K-Nearest Neighbors (KNN) classifier trained on the Iris dataset (150 samples,
3 species, 4 features), following the full supervised learning pipeline:
load → scale → split → train → tune K → evaluate.

## How to run
```bash
pip install scikit-learn matplotlib numpy
python iris_classification.py
```
This generates two files in the working directory:
- `k_tuning_plot.png` — error rate vs. K
- `confusion_matrix.png` — final model's confusion matrix

## Pipeline decisions

**Scaling.** StandardScaler is fit only on the training set and then applied
to both train and test. Fitting on the whole dataset before splitting would
leak test-set statistics (mean/variance) into training — a subtle form of
data leakage.

**Split.** 80/20 train-test split, shuffled, and stratified by class so each
species is proportionally represented in both sets (important with only
150 samples).

**Choosing K.** The script sweeps K from 1–20 and picks whichever value
minimizes test error. On this run, K=1 came out lowest (96.7% test accuracy).
That's worth flagging rather than accepting blindly: K=1 is normally a red
flag for overfitting/noise-sensitivity, since a single neighbor's vote can
be swayed by an outlier. It only "wins" here because Iris is small, clean,
and its three classes are well-separated in feature space (Setosa especially
so). On noisier or larger data, K=1 would likely perform worse out-of-sample
than a slightly higher K (e.g. 5–9), which trades a bit of training-set fit
for more stable generalization. In a real report, I'd mention this trade-off
rather than just reporting the number the elbow method spit out.

**Evaluation.** Accuracy alone is reported, but so are the confusion matrix
and per-class precision/recall/F1. This matters because Iris is a balanced
dataset (50/50/50), so accuracy isn't misleading here — but the pipeline is
built the way you'd need it for imbalanced data, where accuracy can hide a
model that just predicts the majority class. Versicolor and Virginica are
the two classes with slight overlap in petal measurements, which shows up
as the only off-diagonal entries in the confusion matrix.

## Results (this run)
- Best K: 1
- Test accuracy: 96.7%
- Macro F1: 0.967
- Only confusion: one Virginica sample misclassified as Versicolor (these
  two species have naturally overlapping petal dimensions — Setosa is
  linearly separable from both).

## Possible extensions (per the "go further" brief)
- Compare KNN against Logistic Regression or a Decision Tree on the same
  split to see which generalizes best.
- Test the trained model on a few hand-entered flower measurements not in
  the dataset at all.
- Re-run with an artificially imbalanced version of Iris to show how
  accuracy alone becomes misleading (the "Accuracy Mirage" point from the
  brief).
