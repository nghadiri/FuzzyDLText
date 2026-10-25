# Usage

## Requirements

- A Google account with Google Colab and Google Drive. The notebooks mount Drive and use a GPU runtime for fine-tuning.
- Python packages, installed by the notebooks themselves: `transformers`, `emoji`, `Unidecode`, plus `torch`, `scikit-learn`, `pandas`, `numpy`.
- Choquet notebooks: the `ChI` module and `cvxopt`.
- Rule-based notebook: the `fuzzy_system` package (`FuzzySystem`, `FuzzyInputVariable`, `FuzzyOutputVariable`).

`ChI` and `fuzzy_system` are not included in this repository; see the open issue on missing helper modules.

## Prepare the data

1. Take tweet IDs from the [COVID-19 Tweets dataset](https://github.com/lopezbec/COVID19_Tweets_Dataset). The paper used English tweets from July 2020 to January 2021 that met the like, retweet and length criteria.
2. Hydrate the IDs (the paper used TWARC) and label each tweet positive, negative or neutral.
3. Apply the pre-processing steps described in [how-it-works.md](how-it-works.md).
4. Save the splits as CSV files in your Drive working directory (default `My Drive/SEN_BERT`, set in the first cell of each notebook):

| File | Used for |
|------|----------|
| `train.csv` | Fine-tuning (concatenated with validation data for 5-fold cross-validation) |
| `valid.csv` | Validation scores, used to learn the Choquet fuzzy measure |
| `test.csv` | Final evaluation (`test_2.csv` in the rule-based notebook) |

Required columns: `full_text` (the pre-processed tweet) and `Sentiment_Label` (one of the three classes). The rule-based notebook also reads `word_count` and `jacc_sim` (Jaccard similarity to the Covid keyword set) from its result file.

## Run

1. Open a notebook from `src/` in Colab (the badge in its first cell does this).
2. Run it top to bottom. Each notebook fine-tunes BERT, CT-BERT and RoBERTa with these settings:

| Setting | Value |
|---------|-------|
| Max sequence length | 40 |
| Batch size | 16 |
| Epochs | 5 (with early stopping) |
| Cross-validation | 5 folds |
| Seed | 42 |

3. The last cells run the fusion step and print a classification report.

## Which notebook?

| Notebook | Fusion |
|----------|--------|
| `Final_CHi_Sentiment_V7_3class.ipynb` | Choquet integral, fuzzy measure learned from validation scores |
| `Final_densities_CHi_Sentiment_V7_3class.ipynb` | Choquet integral, fuzzy measure from per-model densities |
| `Final_rulebase_Sentiment_V7_3class.ipynb` | Fuzzy rule-based weighting using tweet length and Covid similarity |
