# main.py  —  entry point
# Run order: config → preprocess → bn → mrf → nb → vae
#
# Usage (Colab):
#   !python main.py
#
# Or run cell-by-cell by importing each module manually.

# main.py
# ── 1. Config (imports) ───────────────────────────────────────────────────────
from config import *
import preprocess
from models import bn, mrf, nb, vae

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
import preprocess
df, train_df, test_df, label_encoders, X = preprocess.run()

# ── 3. Bayesian Network ───────────────────────────────────────────────────────
from models import bn
bn_model = bn.run(df, train_df, test_df)

# ── 4. Markov Random Field ────────────────────────────────────────────────────
from models import mrf
mrf_model = mrf.run(df, train_df, test_df)

# ── 5. Naive Bayes ────────────────────────────────────────────────────────────
from models import nb
nb_model = nb.run(df, train_df, test_df, label_encoders)

# ── 6. VAE ────────────────────────────────────────────────────────────────────
from models import vae
vae_model = vae.run(df, train_df, test_df, label_encoders, X)

print("\n" + "="*60)
print("  ALL MODELS DONE")
print("="*60)
