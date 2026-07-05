# Vaccination Tweet Sentiment Analysis

Sentiment analysis of tweets about COVID-19 vaccines (Pfizer/BioNTech, AstraZeneca, Moderna, SputnikV), covering the initial vaccine rollout period (Dec 2020).

## Folder structure

```
NLP/
├── README.md
├── requirements.txt
├── data/
│   └── vaccination_tweets.csv
└── notebooks/
    └── Vaccination_Tweet_Sentimental_Analysis.ipynb
```

## Dataset

`data/vaccination_tweets.csv` — 11,020 tweets, 16 columns:

| Column | Description |
|---|---|
| `id`, `date` | Tweet identifier and timestamp |
| `user_name`, `user_location`, `user_description`, `user_created` | Author metadata |
| `user_followers`, `user_friends`, `user_favourites`, `user_verified` | Author stats |
| `text` | Tweet content (used for analysis) |
| `hashtags`, `source` | Hashtags used, client app |
| `retweets`, `favorites`, `is_retweet` | Engagement metrics |

Only `text` is used for modeling; all other columns are dropped early in the pipeline.

## Pipeline (`notebooks/Vaccination_Tweet_Sentimental_Analysis.ipynb`)

1. **Load & inspect** — read CSV, check shape/nulls/dtypes.
2. **Reduce to text** — drop all metadata columns.
3. **Clean text** — lowercase, strip URLs/punctuation, tokenize (NLTK), remove stopwords.
4. **Deduplicate** — drop repeated tweet text (11,020 → 10,543 rows).
5. **Lemmatize** — reduce words to root form (e.g. `folks` → `folk`, `years` → `year`).
6. **Label sentiment** — polarity score via TextBlob, bucketed into `Negative` (<0), `Neutral` (=0), `Positive` (>0). These labels are a heuristic, not human annotation.
7. **EDA** — sentiment distribution (count/pie chart), word clouds per sentiment class.
8. **Vectorize** — TF-IDF over unigrams + bigrams (~76k features).
9. **Train/test split** — 80/20.
10. **Model & tune** — Logistic Regression and LinearSVC, each hyperparameter-tuned via `GridSearchCV` (scored on macro-F1, not accuracy, so the search doesn't undo class-imbalance handling).

## Results

Best model: **LinearSVC, tuned (`C=10`), TF-IDF features, `class_weight='balanced'`**

| Metric | Value |
|---|---|
| Test accuracy | 85.6% |
| Negative recall | 34% |
| Neutral recall | 98% |
| Positive recall | 85% |

## Known limitations

- **Labels are heuristic, not ground truth.** Sentiment comes from TextBlob polarity applied to the cleaned text, not human annotation — accuracy reflects agreement with TextBlob, not real-world sentiment.
- **Negative class is hard to predict.** It's both the smallest class (~11% of test set) and the noisiest label (TextBlob is weak on sarcasm/negation/political phrasing common in these tweets). `class_weight='balanced'` and macro-F1-scored tuning were applied but didn't meaningfully lift Negative recall — the bottleneck is the label/feature signal, not the classifier's weighting.
- Next steps to actually improve minority-class performance would require either oversampling (e.g. SMOTE) or a hand-labeled validation subset to check whether TextBlob's labels are the limiting factor.

## Environment

Python 3.9+ (tested on 3.13). Package versions pinned in `requirements.txt`:

```
numpy==2.5.0
pandas==2.3.3
matplotlib==3.11.0
seaborn==0.13.2
nltk==3.9.4
textblob==0.20.0
wordcloud==1.9.6
scikit-learn==1.9.0
jupyter, nbconvert, ipykernel   # only needed to run/re-execute the notebook
```

NLTK also needs corpora downloaded separately (not via pip): `punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4` — used for tokenization, stopword removal, and lemmatization.

## Setup

Create an isolated virtual environment (`.venv/`, already git-ignored), then install dependencies into it:

```bash
python -m venv .venv

# activate it
source .venv/Scripts/activate   # Windows (Git Bash)
.venv\Scripts\activate          # Windows (cmd/PowerShell)
source .venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ('punkt','punkt_tab','stopwords','wordnet','omw-1.4')]"
```

## Running the notebook

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace Vaccination_Tweet_Sentimental_Analysis.ipynb
```
