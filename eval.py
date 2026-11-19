from verlens import detect, get_context, vector_store

# q = question, ver = version the question is about (None if not mentioned), kw = word that must be in the docs
qs = [
    # exist only in 1.9 (new after 1.5)
    {"q": "What does FrozenEstimator do in sklearn 1.9?", "ver": "1.9", "kw": "FrozenEstimator"},
    

    # exist only in 1.5 (removed or replaced later)
    
    {"q": "What is the multi_class parameter of LogisticRegression in version 1.5?", "ver": "1.5", "kw": "multi_class"},
   
    # in both (new in 1.5, still there in 1.9)
    {"q": "How does TunedThresholdClassifierCV work in sklearn 1.5?", "ver": "1.5", "kw": "TunedThresholdClassifierCV"},
    {"q": "What is the covariance_eigh solver in PCA in v1.9?", "ver": "1.9", "kw": "covariance_eigh"},
    {"q": "What does d2_log_loss_score compute in sklearn 1.5?", "ver": "1.5", "kw": "d2_log_loss_score"},

    # no version in the question -> router should fall back to 1.9
    {"q": "How do I use TunedThresholdClassifierCV?", "ver": None, "kw": "TunedThresholdClassifierCV"},
    {"q": "What is FrozenEstimator?", "ver": None, "kw": "FrozenEstimator"},
    {"q": "How does HistGradientBoostingClassifier handle categorical features?", "ver": None, "kw": "categorical_features"},
    {"q": "What does GridSearchCV do?", "ver": None, "kw": "GridSearchCV"},

    # numbers that are NOT versions -> router must not get fooled
    {"q": "What does alpha=1.5 do in Ridge?", "ver": None, "kw": "alpha"},
    {"q": "Why does KMeans take 1.5 times longer with more clusters?", "ver": None, "kw": "KMeans"},

        # vague wording: keyword NOT in the question, so retrieval has to find it by meaning
    {"q": "How do I stop a fitted model from being refit inside a pipeline in sklearn 1.9?", "ver": "1.9", "kw": "FrozenEstimator"},
    {"q": "How can I change the decision threshold of a classifier using cross-validation in v1.5?", "ver": "1.5", "kw": "TunedThresholdClassifierCV"},
    {"q": "Which PCA solver is fastest when there are many more samples than features in 1.9?", "ver": "1.9", "kw": "covariance_eigh"},
    {"q": "How do I tell gradient boosting which columns are categorical?", "ver": None, "kw": "categorical_features"},
    {"q": "How do I pick the best hyperparameters by trying every combination?", "ver": None, "kw": "GridSearchCV"},
    {"q": "Which LogisticRegression option picks binary vs multinomial vs ovr in sklearn 1.5?", "ver": "1.5", "kw": "multi_class"},
    {"q": "What metric measures fraction of log loss explained, like R2 but for classifiers, in 1.5?", "ver": "1.5", "kw": "d2_log_loss_score"},

    # harder router: different version styles
    {"q": "scikit-learn v1.5: how does GridSearchCV pick the best params?", "ver": "1.5", "kw": "GridSearchCV"},
    {"q": "I'm on sklearn==1.9, how does GridSearchCV work?", "ver": "1.9", "kw": "GridSearchCV"},
    {"q": "Explain GridSearchCV as of version 1.5", "ver": "1.5", "kw": "GridSearchCV"},

    # router traps: 1.5 / 1.9 that are NOT versions, so should fall back to 1.9
    {"q": "I set C=1.5 in SVC, what does C control?", "ver": None, "kw": "SVC"},
    {"q": "What does StandardScaler do to data with values like 1.5 and 1.9?", "ver": None, "kw": "StandardScaler"},
    {"q": "Why is my model's R2 only 1.5 percent?", "ver": None, "kw": "r2"},
]

# test 1: router (none -> falls back to 1.9)
r_ok = 0
for x in qs:
    want = x["ver"] or "1.9"
    got = detect.invoke({"question": x["q"]})
    if got == want:
        r_ok += 1
    else:
        print("router wrong:", x["q"], "| got", got, "| want", want)
print(f"router acc: {r_ok}/{len(qs)}")

# test 2: retrieval (keyword in top-5 chunks)
h_ok = 0
for x in qs:
    ctx = get_context(x["q"], x["ver"] or "1.9").lower()
    if x["kw"].lower() in ctx:
        h_ok += 1
    else:
        print("retrieval miss:", x["q"])
print(f"retrieval hit: {h_ok}/{len(qs)}")

# sanity: is the keyword even in that version's docs? (if not, the question is bad, not the app)
txt = {}
for v in ["1.5", "1.9"]:
    txt[v] = " ".join(vector_store.get(where={"version": v})["documents"]).lower()

for x in qs:
    v = x["ver"] or "1.9"
    if x["kw"].lower() not in txt[v]:
        print("bad question, kw not in", v, "docs:", x["kw"])