# Project History

This documents how the project evolved, phase by phase, and *why* each change was made. `README.md` describes the project as it stands today; this file explains how it got there — useful if you want to understand the reasoning behind a decision, or just pick up the thread after time away.

## Phase 0 — Starting point

The project began as a single Jupyter notebook (cloned from a public example) plus the raw `vaccination_tweets.csv` dataset, sitting flat in one folder. The original notebook already had a full pipeline: load the CSV, clean the tweet text, score sentiment with TextBlob, do some EDA (count plot, pie chart, word clouds), vectorize with `CountVectorizer`, and train/tune Logistic Regression and LinearSVC with `GridSearchCV`. It worked, but had never been audited — several real bugs turned out to be hiding in it (see Phase 2).

## Phase 1 — Project restructuring

**What changed**: moved the flat layout into a conventional structure — `data/` for the CSV, `notebooks/` for the notebook — and added `.gitignore`, `requirements.txt`, and an isolated `.venv/`. Also caught and fixed an accidental double-nested folder (`Vaccination_Tweet_Sentimental_Analysis/NLP/...`) left over from how the repo was originally cloned, flattening it so the git repo root and the working directory root are the same thing.

**Why**: a flat folder mixing code, data, and docs doesn't scale, and isn't how any real data science project is organized. This is standard practice (separate raw data from code) done early, before more work piled on top of a messy structure.

## Phase 2 — Bug audit and fixes (first pass)

An audit of the original notebook's logic (not just "does it run," but "is the logic actually correct") surfaced two real bugs, both silently corrupting results:

1. **Stopword removal was deleting negation.** NLTK's default English stopword list includes `not`, `no`, `nor`. The cleaning step removed them *before* TextBlob scored sentiment — so `"this vaccine is not safe"` became `"vaccine safe"` before scoring, flipping a clearly negative tweet to score as positive. Verified directly: TextBlob scores `"this vaccine is not safe"` at **-0.25**, but `"vaccine safe"` (the same tweet after the bug) at **+0.50**. Fixed by excluding negation words from the stopword set.
2. **The `@mention`-stripping regex was broken.** It was written as `r'\@w+\#'`, which matches `@` + literal `w` characters + `#` — never actually matching real `@username` patterns. Usernames were silently surviving into the TF-IDF vocabulary as noise. Fixed to `r'@\w+'`.

**Also done in this phase**: the notebook was annotated end-to-end with markdown cells explaining the reasoning behind every step (why TF-IDF, why `class_weight='balanced'`, why `scoring='f1_macro'` in the grid search, etc.), and a known bug in an intermediate cell (`classification_report(y_pred, y_test)` — arguments reversed from scikit-learn's expected order) was flagged.

**Measured impact**: after fixing both bugs, Negative-class recall improved — 39%→42% for tuned Logistic Regression, 34%→38% for tuned LinearSVC — confirming these weren't just theoretical concerns.

## Phase 3 — Real data-leakage fix (the biggest single fix)

A deeper audit — prompted by asking "are these fixes really enough?" — found something more serious: **the TF-IDF vectorizer was being fit on the *entire* dataset before the train/test split.**

```python
# before (leakage):
vect = TfidfVectorizer(...).fit(text_df['text'])       # fit on ALL rows, including test rows
X = vect.transform(text_df['text'])
x_train, x_test, ... = train_test_split(X, Y, ...)      # split happens AFTER
```

This meant the vectorizer's vocabulary and IDF weights were computed using test-set documents too — information the model shouldn't have access to before evaluation. The fix: split the raw text *first*, then fit the vectorizer on training text only.

**Measured impact**: this was the single biggest improvement in the whole project. Negative recall jumped from ~38-42% to **51%** on both models, and accuracy rose ~2 points. Counterintuitively, fixing a leakage bug usually *lowers* reported performance (since the model can no longer cheat) — here it *improved* results, likely because the smaller, training-only vocabulary (58K features vs. the earlier 76K) generalizes better than one partly shaped around memorizing test-set-specific rare terms.

**Also fixed in this phase**: the reversed `classification_report` argument order (from Phase 2's flagged-but-unfixed bug), and removed two redundant `!pip install` cells now that `requirements.txt` existed.

## Phase 4 — Extended analysis

With the core pipeline audited and trustworthy, four extensions were added, each answering a question the core pipeline didn't touch:

1. **Vaccine-brand comparison**: tweets matched to a specific vaccine (Pfizer/BioNTech, Moderna, AstraZeneca, SputnikV) via hashtag + text keyword search, restricted to single-brand-mention tweets for unambiguous attribution. Found Pfizer/BioNTech (n=6,167) has the most favorable sentiment split; AstraZeneca (n=122) skews more negative, plausibly tied to its 2021 safety controversy.
2. **Time & engagement analysis**: brought back `date`, `retweets`, `favorites`, `user_verified` (dropped early in the core pipeline). This caught another real inaccuracy: the dataset actually spans **December 2020 to November 2021** (11 months), not just the initial rollout as the notebook's intro previously claimed.
3. **Stronger models & label-reliability check**: added Naive Bayes baselines (`MultinomialNB`, `ComplementNB`) for comparison — both performed far worse than the linear models (`MultinomialNB` gets just 1% Negative recall, since it has no `class_weight` option), confirming that setting was doing real work. More importantly, TextBlob's labels were checked against two independent sentiment tools: **VADER** (58.0% agreement, full corpus) and a **tweet-specific transformer model** (56.7% agreement, 2,500-tweet sample). This is the most important finding in the project — three reputable sentiment tools disagree with each other on over 40% of these tweets, meaning the ~51% Negative-recall ceiling is a *labeling* problem, not something further model tuning can fix.
4. **Engineering hygiene**: wrapped the best model in an `sklearn.Pipeline`, added 5-fold stratified cross-validation (mean macro-F1 0.788, std 0.011 — a stable estimate), and persisted the final model to `models/sentiment_pipeline.joblib`. A sanity check on the saved model, run as a self-audit rather than just assumed to work, caught a genuine limitation: it mispredicts `"vaccine not safe"` as Positive, because the word `"safe"` has a very strong learned positive coefficient that a rare `"not safe"` bigram can't override — a real, honestly-documented limitation of bag-of-words models, not a bug to paper over.

Every one of these results was independently re-derived from raw data in a separate verification script and diffed against the notebook's outputs — full agreement, no discrepancies.

## Phase 5 — Split into two notebooks

The single notebook had grown to 18 sections / ~90 cells, mixing genuinely distinct concerns: a core modeling pipeline, several fairly independent exploratory extensions, and an engineering/deployment section. That's the point at which standard practice (e.g. the cookiecutter-data-science convention) recommends splitting by concern rather than growing one file indefinitely — partly for readability, partly because re-running the whole thing (including several minutes of transformer inference) just to tweak one small section is wasteful.

**What changed**: split into `01_core_pipeline.ipynb` (Sections 1-13: clean, label, split, vectorize, train, tune) and `02_extended_analysis.ipynb` (Sections 14-18: the four extensions above). Rather than duplicating the cleaning/labeling code in both notebooks, `01_core_pipeline.ipynb` saves its cleaned+labeled data to `data/processed/labeled_tweets.csv`; `02_extended_analysis.ipynb` loads that file and deterministically reproduces the same train/test split and TF-IDF vectorization (same `random_state=42`, same vectorizer settings) rather than re-running text cleaning and TextBlob scoring from scratch.

**A small bug caught during the split**: one tweet's cleaned text was an empty string, which round-tripped through the CSV save/load as `NaN` rather than `""`, breaking the vectorizer on reload. Fixed with an explicit `.fillna('')` after loading — a good example of why re-verifying after a structural change matters, not just assuming a refactor is behavior-preserving.

Both split notebooks were re-executed end-to-end and every result checked against the pre-split single-notebook version — identical, down to the exact decimal.

## Where things stand now

See `README.md` for the current state: folder structure, full results tables, and known limitations. The project's own conclusion (Section 18 of `02_extended_analysis.ipynb`) is that further progress isn't a modeling problem anymore — it's a labeling problem. The natural next step, if this project continues, is to hand-label a validation subset of tweets to determine which sentiment source (TextBlob, VADER, or the transformer) actually tracks human judgment best, or to retrain against the transformer's labels directly instead of TextBlob's.
