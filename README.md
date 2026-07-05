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
8. **Vectorize** — TF-IDF over unigrams + bigrams (~76k features).
9. **Train/test split** — 80/20.
10. **Model & tune** — Logistic Regression and LinearSVC, both trained with `class_weight='balanced'` (to counter the Neutral-class majority) and hyperparameter-tuned via `GridSearchCV` scored on macro-F1 rather than accuracy, so the search doesn't undo the class-imbalance handling by optimizing for the majority class.

The notebook itself now has a markdown explanation before every step — read it directly for the full reasoning behind each choice.

## Results

| Model | Accuracy | Negative recall | Neutral recall | Positive recall |
|---|---|---|---|---|
| Logistic Regression (tuned, `C=10`) | 83.7% | **42%** | 96% | 81% |
| LinearSVC (tuned, `C=10`) | **85.0%** | 38% | 97% | 84% |

Tuned LinearSVC has the highest overall accuracy; tuned Logistic Regression actually catches more true Negatives. Which one is "best" depends on whether overall accuracy or minority-class recall matters more for your use case.

**Effect of fixing the negation/mention bugs** (see Known limitations below): compared to the same pipeline *before* those fixes, Negative recall rose from 39%→42% (LR) and 34%→38% (SVC), at a cost of roughly half a point of overall accuracy — a real, measurable, but modest improvement.

## Known limitations

- **Two real bugs existed in text cleaning and are now fixed**: (1) NLTK's stopword list removes `not`/`no`/`nor`, which was deleting negation before sentiment scoring and flipping polarity on negated tweets (verified: `"this vaccine is not safe"` scored -0.25, but after stopword-stripping the same text became `"vaccine safe"` scoring +0.50); (2) the `@mention`-stripping regex was malformed and never matched real mentions, letting usernames leak into the TF-IDF vocabulary as noise. Both are fixed in the current notebook — see the Results section above for the measured impact.
- **Labels are heuristic, not ground truth.** Sentiment comes from TextBlob polarity applied to the cleaned text, not human annotation — accuracy reflects agreement with TextBlob, not real-world sentiment.
- **Negative class is still the hardest to predict**, even after the fixes. It's both the smallest class (~11% of test set) and the noisiest label (TextBlob is weak on sarcasm/negation/political phrasing common in these tweets). The bug fixes helped, but `class_weight='balanced'` and macro-F1-scored tuning alone can't close the gap further — the remaining bottleneck is the label/feature signal, not the classifier's weighting.
- Next steps to actually improve minority-class performance further: swap TextBlob for VADER (built for social-media text), bring back `date`/`retweets`/`favorites`/`user_verified` for richer EDA, or hand-label a validation subset to check whether TextBlob's labels are the real limiting factor.

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
