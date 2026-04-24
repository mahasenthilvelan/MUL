import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Clean text
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# -----------------------------
# Prepare dataset
# -----------------------------
def prepare_data(df):
    if 'comment_text' in df.columns:
        df['Text'] = df['comment_text']
    if 'toxic' in df.columns:
        df['label'] = df['toxic']

    if 'UserId' not in df.columns:
        df['UserId'] = ['user_' + str(i % 1000) for i in range(len(df))]

    df['clean_text'] = df['Text'].apply(clean_text)

    return df[['Text','label','UserId','clean_text']]

# -----------------------------
# Train baseline
# -----------------------------
def train_model(df):
    X = df['clean_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=1500)

    X_train_tf = tfidf.fit_transform(X_train)
    X_test_tf = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tf, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_tf))

    return model, tfidf, acc

# -----------------------------
# Unlearning
# -----------------------------
def unlearn(df, selected_users):
    user_data = df[df['UserId'].isin(selected_users)]
    remaining = df[~df['UserId'].isin(selected_users)]

    X = remaining['clean_text']
    y = remaining['label']

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    tfidf_u = TfidfVectorizer(max_features=1500)

    X_tr_tf = tfidf_u.fit_transform(X_tr)
    X_te_tf = tfidf_u.transform(X_te)

    model = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=0.05
    )

    model.fit(X_tr_tf, y_tr)

    return model, tfidf_u

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(base_model, tfidf, un_model, tfidf_u, df, selected_users):

    user_data = df[df['UserId'].isin(selected_users)]
    X_user = user_data['clean_text']

    prob_before = base_model.predict_proba(tfidf.transform(X_user))[:,1]
    prob_after = un_model.predict_proba(tfidf_u.transform(X_user))[:,1]

    pred_before = base_model.predict(tfidf.transform(X_user))
    pred_after = un_model.predict(tfidf_u.transform(X_user))

    prediction_change = np.mean(pred_before != pred_after)
    confidence_drop = np.mean(np.abs(prob_before - prob_after))

    # noise
    noise = np.random.laplace(0,0.2,prob_after.shape)
    prob_after_noisy = np.clip(prob_after+noise,0,1)

    attack_X = np.concatenate([prob_before, prob_after_noisy]).reshape(-1,1)
    attack_y = np.concatenate([
        np.ones(len(prob_before)),
        np.zeros(len(prob_after_noisy))
    ])

    rf = RandomForestClassifier()
    rf.fit(attack_X, attack_y)

    auc = roc_auc_score(attack_y, rf.predict_proba(attack_X)[:,1])

    score = (
        0.3 * prediction_change +
        0.2 * min(confidence_drop * 2, 1) +
        0.3 * 0.5 +   # simplified MIA
        0.2 * (1 - abs(auc - 0.5)*2)
    )

    return prediction_change, confidence_drop, auc, score
