# Sports Information Retrieval + Machine Learning

## Project title
**Sports News Information Retrieval and Performance Comparison of KNN, Multinomial Naive Bayes and Logistic Regression**

## Aim
Build a sports-focused Information Retrieval/NLP system that:
1. Converts sports/news documents into TF-IDF vectors.
2. Trains three machine-learning algorithms.
3. Compares accuracy, precision, recall, F1-score and execution time.
4. Ranks the algorithms based on measured performance.
5. Demonstrates information retrieval by returning the most relevant news documents for a user query.

## Algorithms
### 1. K-Nearest Neighbors (KNN)
Finds the most similar training documents and uses their classes to make a prediction.

### 2. Multinomial Naive Bayes
Uses word-feature probabilities to classify documents. It is a common baseline for text classification.

### 3. Logistic Regression
Learns feature weights and predicts the most likely document category.

## Datasets

### Dataset 1 — Sports 20 Newsgroups
Uses the two sports categories available in the standard 20 Newsgroups dataset:
- rec.sport.baseball
- rec.sport.hockey

This makes the first experiment strongly sports-focused.

### Dataset 2 — BBC News
Uses the BBC News text dataset containing five news categories:
- business
- entertainment
- politics
- sport
- tech

This second experiment checks whether the same algorithms work when sports is one category among several news topics.

The 20 Newsgroups dataset contains around 18,000 documents across 20 topics and can be loaded with scikit-learn. The selected sports categories are `rec.sport.baseball` and `rec.sport.hockey`.

A commonly distributed BBC News dataset contains 2,225 articles across five categories including Sport.

## Information Retrieval component
The project also contains a simple search demonstration:
- TF-IDF represents the documents.
- A user enters a query such as:
  `football championship team`
- Cosine similarity compares the query with every document.
- The top 5 most similar documents are returned.

This makes the project an IR project rather than only a classification project.

## Project structure

```text
Sports_IR_ML_Project/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── bbc-text.csv              # downloaded automatically/optional
└── results/
    ├── comparison CSV files
    ├── accuracy graphs
    └── confusion matrices
```

## Windows setup

Open the project folder in VS Code terminal:

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

The first experiment uses the 20 Newsgroups sports categories and may require internet access the first time.

The second experiment tries to download `bbc-text.csv` automatically. If automatic download does not work, download the BBC dataset manually and put `bbc-text.csv` beside `main.py`.

## Output

The `results` folder contains:
- accuracy comparison graphs
- confusion matrices
- CSV performance tables

The terminal prints:
- accuracy
- precision
- recall
- F1-score
- training time
- prediction time
- algorithm ranking

## Suggested report format

1. **Title**
2. **Abstract**
3. **Introduction**
4. **Problem Statement**
5. **Objectives**
6. **Dataset Description**
7. **Data Preprocessing**
8. **TF-IDF Feature Extraction**
9. **Algorithms**
10. **Information Retrieval Method**
11. **Experimental Setup**
12. **Results**
13. **Comparison and Ranking**
14. **Discussion**
15. **Conclusion**
16. **References**

## Suggested problem statement

Sports organizations and news platforms generate a large amount of textual information every day. Manually organizing and retrieving relevant sports news is time-consuming. This project develops a text-based Information Retrieval and classification system that represents documents using TF-IDF, classifies them using three machine-learning algorithms, compares their performance, and retrieves the most relevant documents for a user query.

## Suggested objectives

- To preprocess sports and news text data.
- To represent documents using TF-IDF.
- To implement KNN, Multinomial Naive Bayes and Logistic Regression.
- To compare the algorithms using standard evaluation metrics.
- To rank the algorithms based on experimental results.
- To implement a simple sports/news document retrieval feature.

## Important

Do not manually enter expected accuracy values. Run the program and use the actual generated values in the report. The best algorithm can differ depending on the dataset and preprocessing.
