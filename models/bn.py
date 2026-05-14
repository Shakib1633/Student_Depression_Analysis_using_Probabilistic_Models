# models/bn.py  —  Bayesian Network

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import *

def run(df, train_df, test_df):
    print("\n" + "="*60)
    print("  BAYESIAN NETWORK")
    print("="*60)

    # ── Structure learning ────────────────────────────────────────────────────
    forbidden = [
        ("Depression", "Family History of Mental Illness"),
        ("Depression", "Academic Pressure"),
        ("Depression", "Financial Stress"),
        ("Depression", "Gender"),
        ("Depression", "Age"),
        ("Depression", "Study Satisfaction"),
        ("Depression", "CGPA"),
        ("Depression", "Sleep Duration"),
        ("Depression", "Dietary Habits"),
        ("Depression", "Work/Study Hours"),
        ("Dietary Habits", "Gender"),
        ("Have you ever had suicidal thoughts ?", "Age"),
        ("Have you ever had suicidal thoughts ?", "Family History of Mental Illness"),
        ("Have you ever had suicidal thoughts ?", "Gender"),
        ("CGPA", "Gender"),
        ("Academic Pressure", "Gender"),
        ("Financial Stress", "Gender"),
        ("Sleep Duration", "Gender"),
        ("Study Satisfaction", "Gender"),
        ("Work/Study Hours", "Gender"),
    ]
    required = [
        ("Sleep Duration", "Gender"),
        ("Have you ever had suicidal thoughts ?", "Depression"),
        ("Financial Stress", "Depression"),
        ("Academic Pressure", "Depression"),
    ]

    expert = ExpertKnowledge(forbidden_edges=forbidden, required_edges=required)
    hc = HillClimbSearch(df)
    best_model = hc.estimate(
        scoring_method=BIC(train_df),
        expert_knowledge=expert,
        max_indegree=2,
        max_iter=int(1e4),
        show_progress=True
    )

    print("\nLearned edges:")
    for edge in best_model.edges():
        print(f"  {edge[0]}  →  {edge[1]}")

    # ── Fit CPDs ──────────────────────────────────────────────────────────────
    model = BayesianNetwork(best_model.edges())
    model.fit(train_df)

    cpds = []
    for node in model.nodes():
        estimator = BayesianEstimator(model, train_df)
        cpd = estimator.estimate_cpd(node, prior_type="BDeu", equivalent_sample_size=10)
        cpds.append(cpd)
    model.add_cpds(*cpds)
    print("Model valid:", model.check_model())

    # ── Visualise ─────────────────────────────────────────────────────────────
    net = Network(notebook=False, directed=True, height="700px", width="100%", cdn_resources='remote')
    node_map = {node: str(i+1) for i, node in enumerate(best_model.nodes())}

    for node in best_model.nodes():
        net.add_node(node_map[node], label=node_map[node], title=node,
                    size=40, color='#97c2fc', font={'size': 20})
    for edge in best_model.edges():
        net.add_edge(node_map[edge[0]], node_map[edge[1]], arrows='to')
    net.toggle_physics(False)

    # Build legend HTML and inject into the saved file
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

    net.save_graph("bn_structure.html")
    with open("bn_structure.html", "r") as f:
        html = f.read()
    with open("bn_structure.html", "w") as f:
        f.write(html.replace("</body>", legend_html + "</body>"))

    import webbrowser, os
    webbrowser.open('file://' + os.path.abspath("bn_structure.html"))

    

    # ── Inspect CPDs ──────────────────────────────────────────────────────────
    print("\nCPD for Depression:")
    print(model.get_cpds("Depression"))
    print("\nCPD for CGPA:")
    print(model.get_cpds("CGPA"))
    print("\nCPD for Suicidal thoughts:")
    print(model.get_cpds("Have you ever had suicidal thoughts ?"))

    # ── BIC / Log-likelihood ──────────────────────────────────────────────────
    bic_score = BIC(test_df).score(model)
    num_parameters = sum(
        (cpd.variable_card - 1) * (np.prod(cpd.cardinality[1:]) if len(cpd.variables) > 1 else 1)
        for cpd in model.get_cpds()
    )
    log_likelihood = (bic_score - num_parameters * np.log(len(test_df))) / -2
    print(f"\nLog-Likelihood : {log_likelihood:.4f}")
    print(f"BIC Score      : {bic_score:.4f}")

    # ── Predict ───────────────────────────────────────────────────────────────
    model_input_nodes = [n for n in model.nodes() if n != TARGET]
    X_test = test_df[model_input_nodes]
    y_true = test_df[TARGET]

    y_pred = model.predict(X_test)[TARGET]

    infer = VariableElimination(model)
    y_prob = []
    for _, row in X_test.iterrows():
        evidence = {n: row[n] for n in model_input_nodes}
        q = infer.query(variables=[TARGET], evidence=evidence, show_progress=False)
        y_prob.append(q.values[1] if len(q.values) > 1 else 0)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print(f"\nAccuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"ROC AUC  : {roc_auc_score(y_true, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.title('Confusion Matrix: Depression (BN)')
    plt.tight_layout(); plt.show()

    # ── Inference queries ─────────────────────────────────────────────────────
    infer = VariableElimination(model)

    q1 = infer.query(variables=["Depression"],
                     evidence={"Have you ever had suicidal thoughts ?": 1, "Financial Stress": 4})
    print("\nP(Depression | Suicidal_thoughts=Yes, Financial_Stress=4):")
    print(q1)

    q2 = infer.query(variables=["Depression"],
                     evidence={"Academic Pressure": 5, "Sleep Duration": 3})
    print("\nP(Depression | Academic_Pressure=5, Sleep=LessThan5hrs):")
    print(q2)

    q3 = infer.query(variables=["Have you ever had suicidal thoughts ?"],
                     evidence={"Academic Pressure": 5})
    print("\nP(Suicidal thoughts | Academic_Pressure=5):")
    print(q3)

    q4 = infer.query(variables=["Have you ever had suicidal thoughts ?"],
                     evidence={"Academic Pressure": 2})
    print("\nP(Suicidal thoughts | Academic_Pressure=2):")
    print(q4)

    return model
