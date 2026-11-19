from dotenv import load_dotenv
import time
from verlens import vector_store, ask

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


K = 10
n = len(qs)
hit5, rr, ok = 0, 0.0, 0

for x in qs:
    v = x["ver"] or "1.9"
    kw = x["kw"].lower()

    # retrieval: rank of first chunk containing the keyword
    docs = vector_store.similarity_search(x["q"], filter={"version": v}, k=K)
    rank = next((i + 1 for i, d in enumerate(docs) if kw in d.page_content.lower()), None)
    if rank:
        rr += 1 / rank
        if rank <= 5:
            hit5 += 1
    else:
        print("retrieval miss:", x["q"])

    # end to end: right version picked + answer has the keyword + not "not found"
    r = ask(x["q"])
    a = r["answer"].lower()
    good = r["version"] == v and kw in a and "not found in the docs" not in a
    if good:
        ok += 1
    else:
        print("answer fail:", x["q"], "| ver", r["version"])
    time.sleep(3)  # free-tier rate limit

print(f"hit@5: {hit5}/{n}")
print(f"MRR@{K}: {rr / n:.2f}")
print(f"version-correct answer rate: {ok}/{n}")