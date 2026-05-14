# models/mrf.py  —  Markov Random Field

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import *

def empirical_factor(data, var1, var2, alpha=1.0):
    card1 = int(data[var1].max()) + 1
    card2 = int(data[var2].max()) + 1
    counts = np.zeros((card1, card2))
    for v1, v2 in zip(data[var1], data[var2]):
        counts[v1, v2] += 1
    counts += alpha
    return DiscreteFactor(variables=[var1, var2],
                          cardinality=[card1, card2],
                          values=counts.flatten())

def compute_log_likelihood(model, data):
    ll = 0.0
    factors = model.get_factors()
    for _, row in data.iterrows():
        row_ll = 0.0
        for factor in factors:
            assignment = [int(row[v]) for v in factor.variables]
            val = factor.values[tuple(assignment)]
            row_ll += np.log(val + 1e-10)
        ll += row_ll
    return ll

def run(df, train_df, test_df):
    print("\n" + "="*60)
    print("  MARKOV RANDOM FIELD")
    print("="*60)

    # ── Structure learning ────────────────────────────────────────────────────
    hc = HillClimbSearch(train_df)
    learned_dag = hc.estimate(
        scoring_method=BIC(train_df),
        max_indegree=4,
        max_iter=int(1e4),
        show_progress=True
    )

    undirected_edges = {tuple(sorted([u, v])) for u, v in learned_dag.edges()}

    forbidden_pairs = {tuple(sorted(["Depression", "Family History of Mental Illness"]))}
    required_pairs  = {
        tuple(sorted(["Have you ever had suicidal thoughts ?", "Depression"])),
        tuple(sorted(["Financial Stress", "Depression"])),
        tuple(sorted(["Academic Pressure", "Depression"])),
        tuple(sorted(["Sleep Duration", "Depression"])),
    }
    undirected_edges -= forbidden_pairs
    undirected_edges |= required_pairs

    print("\nFinal MRF edges (undirected):")
    for e in sorted(undirected_edges):
        print(f"  {e[0]}  —  {e[1]}")

    # ── Build & fit ───────────────────────────────────────────────────────────
    mrf = MarkovNetwork(list(undirected_edges))
    factors = [empirical_factor(train_df, u, v) for (u, v) in mrf.edges()]
    mrf.add_factors(*factors)
    print("Model check:", mrf.check_model())

    # ── Visualise ─────────────────────────────────────────────────────────────
    net = Network(notebook=False, directed=False, height="700px", width="100%", cdn_resources='remote')
    node_map = {node: str(i+1) for i, node in enumerate(mrf.nodes())}

    for node in mrf.nodes():
        net.add_node(node_map[node], label=node_map[node], title=node,
                    size=40, color='#f0a500', font={'size': 20})
    for (u, v) in mrf.edges():
        net.add_edge(node_map[u], node_map[v])
    net.toggle_physics(False)

    legend_html = """
    <div style="position:fixed; top:20px; right:20px; background:white;
                border:1px solid #ccc; padding:15px; border-radius:8px;
                font-family:Arial; font-size:13px; z-index:9999;
                max-height:80vh; overflow-y:auto; box-shadow:2px 2px 8px rgba(0,0,0,0.2)">
    <b>Node Legend</b><br><br>
    """
    for name, num in node_map.items():
        legend_html += f"  <b>{num}</b>: {name}<br>"
    legend_html += "</div>"

    net.save_graph("mrf_structure.html")
    with open("mrf_structure.html", "r") as f:
        html = f.read()
    with open("mrf_structure.html", "w") as f:
        f.write(html.replace("</body>", legend_html + "</body>"))

    import webbrowser, os
    webbrowser.open('file://' + os.path.abspath("mrf_structure.html"))

    # ── Inspect factors ───────────────────────────────────────────────────────
    for var in ["Depression", "CGPA", "Have you ever had suicidal thoughts ?"]:
        print(f"\nFactors involving '{var}':")
        for fac in mrf.get_factors():
            if var in fac.variables:
                print(fac)

    # ── BIC / Log-likelihood ──────────────────────────────────────────────────
    ll_score = compute_log_likelihood(mrf, test_df)
    num_parameters = sum((f.cardinality[0]-1)*f.cardinality[1] for f in mrf.get_factors())
    bic_score = -2 * ll_score + num_parameters * np.log(len(test_df))
    print(f"\nLog-Likelihood : {ll_score:.4f}")
    print(f"BIC Score      : {bic_score:.4f}")

    # ── Predict ───────────────────────────────────────────────────────────────
    bp = BeliefPropagation(mrf)
    bp.calibrate()

    evidence_nodes = [n for n in mrf.nodes() if n != TARGET]
    X_test = test_df[evidence_nodes]
    y_true = test_df[TARGET]

    y_pred, y_prob = [], []
    for _, row in X_test.iterrows():
        evidence = {n: int(row[n]) for n in evidence_nodes}
        q = bp.query(variables=[TARGET], evidence=evidence, show_progress=False)
        y_pred.append(int(np.argmax(q.values)))
        y_prob.append(float(q.values[1]) if len(q.values) > 1 else 0.0)

    y_pred, y_prob = np.array(y_pred), np.array(y_prob)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print(f"\nAccuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"ROC AUC  : {roc_auc_score(y_true, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Oranges')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.title('Confusion Matrix: Depression (MRF)')
    plt.tight_layout(); plt.show()

    # ── Inference queries ─────────────────────────────────────────────────────
    q1 = bp.query(variables=["Depression"],
                  evidence={"Have you ever had suicidal thoughts ?": 1, "Financial Stress": 4},
                  show_progress=False)
    print("\nP(Depression | Suicidal_thoughts=Yes, Financial_Stress=4):")
    print(q1)

    q2 = bp.query(variables=["Depression"],
                  evidence={"Academic Pressure": 5, "Sleep Duration": 3},
                  show_progress=False)
    print("\nP(Depression | Academic_Pressure=5, Sleep=LessThan5hrs):")
    print(q2)

    q3 = bp.query(variables=["Have you ever had suicidal thoughts ?"],
                  evidence={"Academic Pressure": 5}, show_progress=False)
    print("\nP(Suicidal thoughts | Academic_Pressure=5):")
    print(q3)

    q4 = bp.query(variables=["Have you ever had suicidal thoughts ?"],
                  evidence={"Academic Pressure": 2}, show_progress=False)
    print("\nP(Suicidal thoughts | Academic_Pressure=2):")
    print(q4)

    return mrf
