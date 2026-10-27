import time
from pathlib import Path
import urllib.request
import zipfile
import io

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MODELS = {
    "KNN": KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute"),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1500, random_state=42)
}

def make_pipeline(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=30000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("classifier", classifier)
    ])

def evaluate(dataset_name, texts, labels, class_names):
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    rows = []
    for name, classifier in MODELS.items():
        model = make_pipeline(classifier)

        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        start = time.perf_counter()
        pred = model.predict(X_test)
        predict_time = time.perf_counter() - start

        accuracy = accuracy_score(y_test, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="weighted", zero_division=0
        )

        rows.append({
            "dataset": dataset_name,
            "algorithm": name,
            "accuracy_percent": round(accuracy * 100, 2),
            "weighted_precision_percent": round(precision * 100, 2),
            "weighted_recall_percent": round(recall * 100, 2),
            "weighted_f1_percent": round(f1 * 100, 2),
            "training_time_seconds": round(train_time, 3),
            "prediction_time_seconds": round(predict_time, 3)
        })

        print("\n" + "=" * 65)
        print(f"{dataset_name} — {name}")
        print("=" * 65)
        print(f"Accuracy : {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall   : {recall * 100:.2f}%")
        print(f"F1-score : {f1 * 100:.2f}%")
        print(f"Train time: {train_time:.3f}s")
        print(classification_report(
            y_test, pred, target_names=class_names, zero_division=0
        ))

        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=class_names,
            xticks_rotation=45, cmap="Blues", colorbar=False, ax=ax
        )
        ax.set_title(f"{dataset_name} — {name}")
        fig.tight_layout()
        safe = dataset_name.lower().replace(" ", "_") + "_" + name.lower().replace(" ", "_")
        fig.savefig(RESULTS / f"{safe}_confusion_matrix.png", dpi=160)
        plt.close(fig)

    df = pd.DataFrame(rows).sort_values(
        ["accuracy_percent", "weighted_f1_percent"], ascending=False
    ).reset_index(drop=True)
    df.insert(0, "rank_by_accuracy", range(1, len(df) + 1))

    safe = dataset_name.lower().replace(" ", "_")
    df.to_csv(RESULTS / f"{safe}_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    df.sort_values("accuracy_percent").plot(
        x="algorithm", y="accuracy_percent", kind="barh",
        legend=False, ax=ax
    )
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Algorithm")
    ax.set_xlim(0, 100)
    ax.set_title(f"Sports IR — Algorithm Accuracy: {dataset_name}")
    fig.tight_layout()
    fig.savefig(RESULTS / f"{safe}_accuracy.png", dpi=160)
    plt.close(fig)

    print("\nRANKING")
    print(df[[
        "rank_by_accuracy", "algorithm", "accuracy_percent",
        "weighted_f1_percent", "training_time_seconds"
    ]].to_string(index=False))
    return df

def sports_20newsgroups():
    categories = ["rec.sport.baseball", "rec.sport.hockey"]
    print("Loading 20 Newsgroups sports categories...")
    train = fetch_20newsgroups(
        subset="train", categories=categories, shuffle=True, random_state=42,
        remove=("headers", "footers", "quotes")
    )
    test = fetch_20newsgroups(
        subset="test", categories=categories, shuffle=True, random_state=42,
        remove=("headers", "footers", "quotes")
    )
    texts = list(train.data) + list(test.data)
    labels = list(train.target) + list(test.target)
    return evaluate("Sports 20 Newsgroups", texts, labels, list(train.target_names))

BBC_URLS = [
    "https://huggingface.co/datasets/SetFit/bbc-news/resolve/main/bbc-text.csv",
    "https://raw.githubusercontent.com/mdsohaib/BBC-News-Classification/master/bbc-text.csv",
]

def download_bbc():
    path = ROOT / "bbc-text.csv"
    if path.exists():
        return path
    print("\nDownloading BBC News dataset...")
    last_error = None
    for url in BBC_URLS:
        try:
            print(f"Trying: {url}")
            urllib.request.urlretrieve(url, path)
            return path
        except Exception as exc:
            last_error = exc
            if path.exists():
                path.unlink()
    print("Automatic BBC download failed:", last_error)
    print("You can download bbc-text.csv manually and place it beside main.py.")
    return None

def bbc_news():
    path = download_bbc()
    if path is None:
        return None
    df = pd.read_csv(path)
    required = {"category", "text"}
    if not required.issubset(df.columns):
        raise ValueError("bbc-text.csv must contain 'category' and 'text' columns.")
    df = df.dropna(subset=["category", "text"])
    return evaluate(
        "BBC News",
        df["text"].astype(str).tolist(),
        df["category"].astype(str).tolist(),
        sorted(df["category"].unique())
    )

def sports_search_demo():
    """Simple IR retrieval demonstration using TF-IDF + cosine similarity."""
    path = ROOT / "bbc-text.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).dropna(subset=["text"])
    vectorizer = TfidfVectorizer(stop_words="english", max_features=30000)
    matrix = vectorizer.fit_transform(df["text"].astype(str))

    query = input("\nEnter a sports search query (or press Enter to skip): ").strip()
    if not query:
        return

    from sklearn.metrics.pairwise import cosine_similarity
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, matrix).ravel()
    top = scores.argsort()[-5:][::-1]

    print("\nTop 5 retrieved documents:")
    for rank, idx in enumerate(top, 1):
        print(f"\n{rank}. Category: {df.iloc[idx]['category']} | Similarity: {scores[idx]:.3f}")
        print(df.iloc[idx]["text"][:350].replace("\n", " ") + "...")

if __name__ == "__main__":
    print("=" * 70)
    print("SPORTS INFORMATION RETRIEVAL + ML ALGORITHM COMPARISON")
    print("=" * 70)
    print("\nExperiment 1: Sports-only classification")
    sports_20newsgroups()

    print("\nExperiment 2: General news classification containing a Sport class")
    bbc_result = bbc_news()

    if bbc_result is not None:
        sports_search_demo()

    print("\nDone. Open the results/ folder for CSV tables, graphs and confusion matrices.")
