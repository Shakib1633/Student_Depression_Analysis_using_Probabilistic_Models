# models/nb.py  —  Naive Bayes (CategoricalNB)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import *

def run(df, train_df, test_df, label_encoders):
    print("\n" + "="*60)
    print("  NAIVE BAYES")
    print("="*60)

    # ── Prepare data ──────────────────────────────────────────────────────────
    X_train = train_df.drop(columns=[TARGET]).values
    y_train = train_df[TARGET].values
    X_test  = test_df.drop(columns=[TARGET]).values
    y_test  = test_df[TARGET].values

    # ── Train ─────────────────────────────────────────────────────────────────
    model = CategoricalNB(alpha=1.0)
    model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"ROC AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.title('Confusion Matrix: Depression (Naive Bayes)')
    plt.tight_layout(); plt.show()

    # ── Single input prediction ───────────────────────────────────────────────
    def predict_depression(gender, age, academic_pressure, cgpa,
                           study_satisfaction, sleep_duration, dietary_habits,
                           suicidal_thoughts, work_study_hours, financial_stress,
                           family_history):
        age_bin   = pd.cut([age],   bins=[0,18,25,35,100],   labels=[0,1,2,3], include_lowest=True).astype(int)[0]
        cgpa_bin  = pd.cut([cgpa],  bins=[-0.01,5.0,6.5,8.0,10.01], labels=[0,1,2,3], include_lowest=True).astype(int)[0]
        hours_bin = pd.cut([work_study_hours], bins=[-0.01,4.5,8.5,10.5,12.01], labels=[0,1,2,3], include_lowest=True).astype(int)[0]

        def encode(col, val):
            return int(label_encoders[col].transform([val])[0])

        raw = np.array([[
            encode('Gender', gender), age_bin, academic_pressure, cgpa_bin,
            study_satisfaction, encode('Sleep Duration', sleep_duration),
            encode('Dietary Habits', dietary_habits),
            encode('Have you ever had suicidal thoughts ?', suicidal_thoughts),
            hours_bin, financial_stress,
            encode('Family History of Mental Illness', family_history)
        ]])

        pred  = model.predict(raw)[0]
        probs = model.predict_proba(raw)[0]
        print(f"\n{'─'*40}")
        print(f"  Prediction       : {'Depressed' if pred == 1 else 'Not Depressed'}")
        print(f"  P(Not Depressed) : {probs[0]:.4f}")
        print(f"  P(Depressed)     : {probs[1]:.4f}")
        print(f"{'─'*40}")
        return pred, probs

    # Example
    predict_depression(
        gender="Male", age=22, academic_pressure=5, cgpa=6.0,
        study_satisfaction=2, sleep_duration="Less than 5 hours",
        dietary_habits="Unhealthy", suicidal_thoughts="Yes",
        work_study_hours=9, financial_stress=4, family_history="No"
    )

    return model
