# Vaccination Tweet Sentiment Analysis

Sentiment analysis of tweets about COVID-19 vaccines (Pfizer/BioNTech, AstraZeneca, Moderna, SputnikV), covering the initial vaccine rollout period (Dec 2020).

## Folder structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
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
3. **Clean text** — lowercase, strip URLs, strip `@mentions`, strip punctuation, tokenize (NLTK), remove stopwords **except negation words** (`not`/`no`/`nor` are deliberately kept — removing them before sentiment scoring flips polarity, e.g. `"not safe"` → `"safe"`).
4. **Deduplicate** — drop repeated cleaned tweet text (11,020 → 10,451 rows).
5. **Lemmatize** — reduce words to root form (e.g. `folks` → `folk`, `years` → `year`).
6. **Label sentiment** — polarity score via TextBlob, bucketed into `Negative` (<0), `Neutral` (=0), `Positive` (>0). These labels are a heuristic, not human annotation.
7. **EDA** — sentiment distribution (count/pie chart), word clouds per sentiment class.
8. **Train/test split** — 80/20, done on the raw cleaned text *before* vectorizing (see below).
9. **Vectorize** — TF-IDF over unigrams + bigrams, fit on the training text only (~58k features).
10. **Model & tune** — Logistic Regression and LinearSVC, both trained with `class_weight='balanced'` (to counter the Neutral-class majority) and hyperparameter-tuned via `GridSearchCV` scored on macro-F1 rather than accuracy, so the search doesn't undo the class-imbalance handling by optimizing for the majority class.

The notebook itself now has a markdown explanation before every step — read it directly for the full reasoning behind each choice.

## Results

| Model | Accuracy | Negative recall | Neutral recall | Positive recall |
|---|---|---|---|---|
| Logistic Regression (tuned, `C=10`) | 85.3% | 51% | 94% | 85% |
| LinearSVC (tuned, `C=10`) | **86.9%** | **51%** | 95% | 87% |

Tuned LinearSVC is the best overall model — highest accuracy and Positive recall, tied with tuned Logistic Regression on Negative recall.

**Effect of fixing three real bugs**, in order of actual impact:

1. **TF-IDF was fit on the full dataset before the train/test split** — a genuine data-leakage bug (test-set word statistics leaked into the vocabulary/IDF weights used for training). Fixing it — splitting the raw text first, then fitting TF-IDF on training text only — was the biggest single improvement: **Negative recall jumped from ~38-42% to 51%** on both models, and accuracy rose ~2 points. Counterintuitively, fixing leakage usually *lowers* reported performance; here it improved it, likely because the smaller training-only vocabulary (58K features vs. the earlier 76K) generalizes better than one partly shaped around test-set-specific rare terms.
2. **Stopword removal was deleting negation** (`not`/`no`/`nor` are NLTK stopwords), flipping polarity on negated tweets before sentiment scoring (verified: `"this vaccine is not safe"` scored -0.25, but after stopword-stripping became `"vaccine safe"` scoring +0.50).
3. **The `@mention`-stripping regex was malformed** and never matched real mentions, letting usernames leak into the TF-IDF vocabulary as noise.

All three are fixed in the current notebook.

## Known limitations

- **Labels are heuristic, not ground truth.** Sentiment comes from TextBlob polarity applied to the cleaned text, not human annotation — accuracy reflects agreement with TextBlob, not real-world sentiment.
- **Negative class is still the hardest to predict**, even after all three fixes. It's both the smallest class (~11% of test set) and the noisiest label (TextBlob is weak on sarcasm/implicit criticism that doesn't use obviously negative words). No amount of further `C`-tuning or class-weighting can teach a linear model a signal that isn't reliably present in the labels it's trained to reproduce.
- Next steps to improve minority-class performance further: swap TextBlob for VADER (built for social-media text), bring back `date`/`retweets`/`favorites`/`user_verified` for richer EDA, add cross-validation and an `sklearn.Pipeline` (harder to accidentally leak again), or hand-label a validation subset to check whether TextBlob's labels are the real limiting factor.

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
