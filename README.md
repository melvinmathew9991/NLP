# Vaccination Tweet Sentiment Analysis

Sentiment analysis of tweets about COVID-19 vaccines (Pfizer/BioNTech, AstraZeneca, Moderna, SputnikV), spanning December 2020 to November 2021.

## Folder structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── vaccination_tweets.csv
├── models/
│   └── sentiment_pipeline.joblib   # persisted, ready to reload with joblib.load()
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

Only `text` is used for the core modeling pipeline; other columns are dropped early on but brought back for the Extended Analysis (see below).

## Core Pipeline (`notebooks/Vaccination_Tweet_Sentimental_Analysis.ipynb`, Sections 1-13)

1. **Load & inspect** — read CSV, check shape/nulls/dtypes.
2. **Reduce to text** — drop all metadata columns.
3. **Clean text** — lowercase, strip URLs, strip `@mentions`, strip punctuation, tokenize (NLTK), remove stopwords **except negation words** (`not`/`no`/`nor` are deliberately kept — removing them before sentiment scoring flips polarity, e.g. `"not safe"` → `"safe"`).
4. **Deduplicate** — drop repeated cleaned tweet text (11,020 → 10,451 rows).
5. **Lemmatize** — reduce words to root form (e.g. `folks` → `folk`, `years` → `year`).
6. **Label sentiment** — polarity score via TextBlob, bucketed into `Negative` (<0), `Neutral` (=0), `Positive` (>0). These labels are a heuristic, not human annotation.
7. **EDA** — sentiment distribution (count/pie chart), word clouds per sentiment class.
8. **Train/test split** — 80/20, done on the raw cleaned text *before* vectorizing (see below).
9. **Vectorize** — TF-IDF over unigrams + bigrams, fit on the training text only (~58k features).
10. **Model & tune** — Logistic Regression and LinearSVC, both trained with `class_weight='balanced'` and hyperparameter-tuned via `GridSearchCV` scored on macro-F1 rather than accuracy.

The notebook has a markdown explanation before every step — read it directly for the full reasoning behind each choice.

## Extended Analysis (Sections 14-18)

Goes beyond the core pipeline using data dropped in Step 2 above (`hashtags`, `date`, `retweets`, `favorites`, `user_verified`):

- **Section 14 — Vaccine-brand comparison**: sentiment split by vaccine brand (Pfizer/Moderna/AstraZeneca/SputnikV), matched via hashtags + text keyword search.
- **Section 15 — Time & engagement**: sentiment trend over the dataset's actual Dec 2020-Nov 2021 span, engagement (retweets/favorites) vs. sentiment, verified vs. unverified accounts.
- **Section 16 — Stronger models & label-reliability check**: Naive Bayes baselines (MultinomialNB, ComplementNB) compared against the linear models, plus an independent check of TextBlob's labels against VADER and a tweet-specific transformer model.
- **Section 17 — Engineering hygiene**: the best model wrapped in an `sklearn.Pipeline`, evaluated with 5-fold stratified cross-validation, and persisted to `models/sentiment_pipeline.joblib`.
- **Section 18 — Final project summary**: ties every finding together.

## Results

| Model | Accuracy | Negative recall | Neutral recall | Positive recall |
|---|---|---|---|---|
| Logistic Regression (tuned, `C=10`) | 85.3% | 51% | 94% | 85% |
| LinearSVC (tuned, `C=10`) | **86.9%** | **51%** | 95% | 87% |
| MultinomialNB | 75.9% | 1% | 91% | 80% |
| ComplementNB | 76.2% | 15% | 83% | 86% |

Tuned LinearSVC is the best overall model. The Naive Bayes variants confirm `class_weight='balanced'` (unavailable on either NB variant) is doing real work — without it, the model barely predicts the minority Negative class at all.

**Effect of fixing three real bugs**, in order of actual impact:

1. **TF-IDF was fit on the full dataset before the train/test split** — a genuine data-leakage bug. Fixing it — splitting the raw text first, then fitting TF-IDF on training text only — was the biggest single improvement: **Negative recall jumped from ~38-42% to 51%** on both models, and accuracy rose ~2 points.
2. **Stopword removal was deleting negation** (`not`/`no`/`nor` are NLTK stopwords), flipping polarity on negated tweets before sentiment scoring.
3. **The `@mention`-stripping regex was malformed** and never matched real mentions, letting usernames leak into the TF-IDF vocabulary as noise.

**Vaccine-brand sentiment** (single-brand-mention tweets only): Pfizer/BioNTech (n=6,167) has the most favorable split (9.2% Negative, 40.1% Positive); AstraZeneca (n=122) skews more negative (13.9%), plausibly tied to its 2021 safety controversy. Moderna (n=52) and SputnikV (n=13) have too few tweets for reliable percentages.

**Is TextBlob's labeling actually reliable?** No — and this is the most important finding in the project. TextBlob agrees with VADER on only **58.0%** of tweets (full corpus, n=10,451), and with a tweet-specific transformer model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) on only **56.7%** (2,500-tweet sample). Three independent, reputable sentiment tools disagree with each other on over 40% of this dataset. This means the ~51% Negative-recall ceiling is fundamentally a **labeling-reliability problem**, not something further model tuning can fix.

**A concrete illustration of why**: the persisted model correctly classifies `"vaccine side effect terrible"` and `"worst experience ever getting vaccine"` as Negative, but gets `"vaccine not safe"` (predicts Positive) and `"not trust vaccine"` (predicts Neutral) wrong. Digging into the model's learned weights: the word `"safe"` alone has a very strong positive-class coefficient (from training contexts like "vaccine is safe and effective"), and the negation-aware fix from Section 3 only helps when a negated phrase is common enough in training to be learned as its own bigram feature — it has no general mechanism for "negation flips the following word." This is a structural limitation of bag-of-words models, not a bug.

## Known limitations

- **Labels are heuristic, not ground truth**, and independently shown to be unstable — see the label-reliability finding above.
- **Negative class is still the hardest to predict**, both because it's the smallest class (~11% of test set) and because bag-of-words models don't compositionally understand negation (see the concrete example above).
- **Next steps**, roughly in order of expected impact: adopt the transformer's labels instead of TextBlob's and retrain against those; hand-label a validation subset to determine which labeling source actually tracks human judgment; or move to a model architecture (e.g. fine-tuning a transformer directly) that captures negation and word order instead of treating text as a bag of independent tokens.

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
joblib==1.5.3
transformers==5.13.0
torch==2.12.1
jupyter, nbconvert, ipykernel   # only needed to run/re-execute the notebook
```

NLTK also needs corpora downloaded separately (not via pip): `punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`, `vader_lexicon`.

**Note on `torch`**: the CPU-only wheel isn't always resolved correctly from a plain `pip install -r requirements.txt` depending on your platform. If it fails or pulls an unexpectedly large CUDA build, install it explicitly first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Setup

Create an isolated virtual environment (`.venv/`, already git-ignored), then install dependencies into it:

```bash
python -m venv .venv

# activate it
source .venv/Scripts/activate   # Windows (Git Bash)
.venv\Scripts\activate          # Windows (cmd/PowerShell)
source .venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ('punkt','punkt_tab','stopwords','wordnet','omw-1.4','vader_lexicon')]"
```

## Running the notebook

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace Vaccination_Tweet_Sentimental_Analysis.ipynb
```

The transformer-based label comparison (Section 16) downloads a ~500MB model from Hugging Face on first run and takes a few minutes to run inference on its 2,500-tweet sample; subsequent runs reuse the cached model weights.

## Reusing the saved model

```python
import joblib
pipeline = joblib.load('models/sentiment_pipeline.joblib')
pipeline.predict(["vaccine side effect terrible", "grateful got shot today"])
```

Note: the saved pipeline expects text already cleaned the same way as training (lowercased, URLs/mentions stripped, lemmatized — see Section 3's `data_processing()` and Section 4's `lemmatizing()`). It does not include that preprocessing step itself, so raw uncleaned tweets should be passed through those functions first.
