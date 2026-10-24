"""Train the demo tamper classifier on synthetic genuine/tampered cards.

    python train_model.py            # writes models/tamper_model.joblib
"""
import os
import random
import sys

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from screening import forensics, synthetic as syn
from screening.ml import FEATURES, MODEL_PATH, vectorise

N = int(sys.argv[1]) if len(sys.argv) > 1 else 160
rng = random.Random(7)
FIRST = ["Aarav", "Priya", "Rohan", "Sneha", "Vikram", "Anjali", "Karan", "Meera", "Arjun", "Divya"]
LAST = ["Sharma", "Patil", "Deshmukh", "Iyer", "Gupta", "Kulkarni", "Reddy", "Joshi", "Nair", "Singh"]


def sample(i):
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    dob = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1965, 2003)}"
    card = syn.make_card(name, dob, syn.valid_aadhaar(rng), syn.valid_pan(rng, name.split()[1]), seed=i)
    return card, dob


X, y = [], []
for i in range(N):
    card, dob = sample(i)
    q = rng.choice([80, 85, 90])
    genuine = syn.to_jpeg(card, q)
    kind = rng.choice(["text", "splice", "clone", "software"])
    if kind == "text":
        bad = syn.to_jpeg(syn.tamper_text(forensics.load_image(syn.to_jpeg(card, 70)), "01/01/1980"), q)
    elif kind == "splice":
        bad = syn.to_jpeg(syn.tamper_splice(forensics.load_image(syn.to_jpeg(card, 75))), q)
    elif kind == "clone":
        bad = syn.to_jpeg(syn.tamper_clone(forensics.load_image(genuine)), q)
    else:
        bad = syn.to_jpeg(syn.tamper_text(forensics.load_image(genuine), "05/05/1985"), q, software="Adobe Photoshop 25.0")
    for data, label in ((genuine, 0), (bad, 1)):
        feats, _, _ = forensics.analyse_image(data)
        X.append(vectorise(feats)[0]); y.append(label)
    if (i + 1) % 20 == 0:
        print(f"  rendered {i + 1}/{N} pairs", flush=True)

X, y = np.array(X), np.array(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)
clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=1)
clf.fit(Xtr, ytr)
p = clf.predict_proba(Xte)[:, 1]
print(classification_report(yte, p > 0.5, target_names=["genuine", "tampered"]))
print("ROC AUC:", round(roc_auc_score(yte, p), 3))
clf.fit(X, y)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(clf, MODEL_PATH)
imp = sorted(zip(FEATURES, clf.feature_importances_), key=lambda t: -t[1])[:5]
print("Top features:", [(k, round(v, 3)) for k, v in imp])
print("Saved", MODEL_PATH)
