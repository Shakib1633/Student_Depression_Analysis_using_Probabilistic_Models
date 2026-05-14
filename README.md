# Student_Depression_Analysis_using_Probabilistic_Models

A comparative study of four probabilistic models for classifying student depression using the **Student Depression Dataset**. Built as part of a Probabilistic Graphical Models (PGM) course project.

---

## Models Implemented

| Model | Type | Library |
|---|---|---|
| Bayesian Network (BN) | Directed graphical model | `pgmpy` |
| Markov Random Field (MRF) | Undirected graphical model | `pgmpy` |
| Tree Augmented Naive Bayes (TAN) | Directed graphical model | `pgmpy` |
| Naive Bayes | Probabilistic classifier | `scikit-learn` |
| Variational Autoencoder (VAE) | Deep generative model | `PyTorch` |

---

## Project Structure

```
depression_pgm/
│
├── main.py              ← Run this to execute all models in order
├── config.py            ← All imports and global constants
├── preprocess.py        ← Data loading, cleaning, binning, encoding
│
└── models/
    ├── bn.py            ← Bayesian Network
    ├── mrf.py           ← Markov Random Field
    ├── nb.py            ← Naive Bayes
    └── vae.py           ← Variational Autoencoder
```

---

## Requirements

- Python 3.8+
- pip

Install all dependencies with:

```bash
pip install pgmpy pyvis torch scikit-learn pandas numpy matplotlib seaborn networkx
```

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/Shakib1633/Student_Depression_Analysis_using_Probabilistic_Models.git
cd depression-pgm
```

**2. Download the dataset**

Download the **Student Depression Dataset** from [Kaggle](https://www.kaggle.com/datasets/hopesb/student-depression-dataset?resource=download) and place it anywhere on your machine.

**3. Update the dataset path**

Open `config.py` and update `DATA_PATH` to point to your dataset:

```python
# Windows
DATA_PATH = r'C:\Users\YourName\Documents\Student Depression Dataset.csv'

# Mac / Linux
DATA_PATH = '/home/yourname/datasets/Student Depression Dataset.csv'
```

---

## Running the Project

Run all models sequentially from the project root:

```bash
python main.py
```

This will execute in the following order:

```
config.py  →  preprocess.py  →  bn.py  →  mrf.py  →  nb.py  →  vae.py
```

Each model prints its metrics (Accuracy, Precision, F1, ROC AUC), classification report, and confusion matrix to the terminal.

---

## Viewing the Graph Visualisations

The Bayesian Network and MRF models generate interactive HTML graph files in your project folder:

```
bn_structure.html   ← Bayesian Network DAG
mrf_structure.html  ← Markov Random Field
```

They open automatically in your browser after running. If they don't, open them manually:

**Windows**
```bash
start bn_structure.html
start mrf_structure.html
```

**Mac**
```bash
open bn_structure.html
open mrf_structure.html
```

**Linux**
```bash
xdg-open bn_structure.html
xdg-open mrf_structure.html
```

Or simply **double-click** the `.html` files in File Explorer / Finder.

Each graph has a **legend panel** in the top-right corner mapping node numbers to feature names.

> **Tip:** You can drag nodes around and zoom in/out inside the browser.

---

## Dataset Features

| Feature | Type | Notes |
|---|---|---|
| Gender | Categorical | Male / Female |
| Age | Continuous → 4 bins | Teen / YoungAdult / Adult / MatureAdult |
| Academic Pressure | Discrete (1–5) | — |
| CGPA | Continuous → 4 bins | Low / Medium / High / VeryHigh |
| Study Satisfaction | Discrete (1–5) | — |
| Sleep Duration | Categorical | Less than 5 hrs / 5–6 hrs / 7–8 hrs / More than 8 hrs |
| Dietary Habits | Categorical | Healthy / Moderate / Unhealthy |
| Suicidal Thoughts | Categorical | Yes / No |
| Work/Study Hours | Continuous → 4 bins | Low / Medium / High / VeryHigh |
| Financial Stress | Discrete (1–5) | — |
| Family History of Mental Illness | Categorical | Yes / No |
| **Depression** | Binary | **Target variable** |

---

## Making a Single Prediction

Each model file has a `predict_depression()` function at the bottom. After running `main.py`, you can call it with raw input values:

```python
predict_depression(
    gender              = "Male",
    age                 = 22,
    academic_pressure   = 5,
    cgpa                = 6.0,
    study_satisfaction  = 2,
    sleep_duration      = "Less than 5 hours",
    dietary_habits      = "Unhealthy",
    suicidal_thoughts   = "Yes",
    work_study_hours    = 9,
    financial_stress    = 4,
    family_history      = "No"
)
```

---

## Notes

- All models share the same preprocessed data and train/test split (`random_state=42`, `test_size=0.2`)
- The BN uses Hill-Climb search with BIC scoring and domain-knowledge constraints (forbidden/required edges)
- The MRF converts the learned DAG skeleton to undirected edges and estimates pairwise factors with Laplace smoothing
- The VAE uses a combined loss: reconstruction + KL divergence + classification
- String columns are label-encoded; continuous columns (Age, CGPA, Work/Study Hours) are binned into 4 discrete categories
