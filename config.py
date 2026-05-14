# config.py  —  all imports and global constants for the project

import warnings
warnings.filterwarnings("ignore")

# ── Standard ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
# import IPython

# ── Sklearn ───────────────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.naive_bayes import CategoricalNB
from sklearn.decomposition import PCA

# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── pgmpy ─────────────────────────────────────────────────────────────────────
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.models import DiscreteMarkovNetwork as MarkovNetwork
from pgmpy.estimators import HillClimbSearch, BIC, BayesianEstimator, StructureScore
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.causal_discovery import ExpertKnowledge

# ── Pyvis ─────────────────────────────────────────────────────────────────────
from pyvis.network import Network
# from IPython.core.display import display, HTML

# ── Global constants ──────────────────────────────────────────────────────────
DATA_PATH = r'C:\Users\User\Downloads\depression_pgm\Student Depression Dataset.csv'
TARGET     = 'Depression'
TEST_SIZE  = 0.2
RANDOM_STATE = 42
