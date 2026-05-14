# preprocess.py  —  load, clean, bin, encode, and split the dataset
# Returns: df, train_df, test_df, label_encoders, X (float32 normalised for VAE)

from config import *

def run():
    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    print("Before:", df.shape, "\n")

    # ── Filter & drop irrelevant columns ─────────────────────────────────────
    df = df[df['Profession'] == 'Student']
    df = df.drop(['City', 'id', 'Profession', 'Work Pressure',
                  'Job Satisfaction', 'Degree'], axis=1)
    df["Financial Stress"] = df["Financial Stress"].fillna(0)

    print("Null values after cleaning:\n", df.isnull().sum(), "\n")

    # ── Bin continuous columns ────────────────────────────────────────────────
    df["Age"] = pd.cut(df["Age"],
                       bins=[0, 18, 25, 35, 100],
                       labels=[0, 1, 2, 3],
                       include_lowest=True).astype(int)

    df["CGPA"] = pd.cut(df["CGPA"],
                        bins=[-0.01, 5.0, 6.5, 8.0, 10.01],
                        labels=[0, 1, 2, 3],
                        include_lowest=True).astype(int)

    df["Work/Study Hours"] = pd.cut(df["Work/Study Hours"],
                                    bins=[-0.01, 4.5, 8.5, 10.5, 12.01],
                                    labels=[0, 1, 2, 3],
                                    include_lowest=True).astype(int)

    # ── Label-encode string columns ───────────────────────────────────────────
    string_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    print(f"String columns to label-encode: {string_cols}\n")

    label_encoders = {}
    for col in string_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"Encoding for {col}:")
        for label, idx in zip(le.classes_, le.transform(le.classes_)):
            print(f"  {label} = {idx}")

    df = df.astype(int)
    print(f"\nFinal shape: {df.shape}")
    print(df.head(), "\n")

    # ── Train / test split ────────────────────────────────────────────────────
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE,
                                         random_state=RANDOM_STATE)

    # ── Float32 normalised array (used by VAE) ────────────────────────────────
    X = df.drop(columns=[TARGET]).values.astype(np.float32)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

    print(f"Train size: {len(train_df)}  |  Test size: {len(test_df)}")
    return df, train_df, test_df, label_encoders, X


if __name__ == "__main__":
    df, train_df, test_df, label_encoders, X = run()
