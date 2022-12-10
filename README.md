# german_credit_python

A Logistic Regression Python model to predict loan default.

## Running Locally

To run this model locally, create a new Python 3.8.2 virtual environment
(such as with `pyenv`). Then, use the following command to update `pip`
and `setuptools`:

```
python3 -m pip install --upgrade setuptools
python3 -m pip install --upgrade pip
```

And install the required libraries:

```
python3 -m pip install -r requirements.txt
```

The main source code is contained in `german_credit.py`. To test all code at-once, run

```
python3 german_credit.py
```

## Details

Model was trained on the German Credit Data dataset.
 - `german_credit_data.csv` is the raw data.
 - `logreg_classifier.pickle` is the trained model artifact.

## Scoring (Inference) Requests

### Sample Inputs

Choose a record (row) from one of the following JSON-lines files:
 - `df_baseline.json`
 - `df_sample.json`

### Sample Output

The output of the scoring request when the input records is the first row of `df_sample.json` is a the following dictionary:
```json
{
    "id": 687,
    "duration_months": 36,
    "credit_amount": 2862,
    "installment_rate": 4,
    "present_residence_since": 3,
    "age_years": 30,
    "number_existing_credits": 1,
    "checking_status": "A12",
    "credit_history": "A33",
    "purpose": "A40",
    "savings_account": "A62",
    "present_employment_since": "A75",
    "debtors_guarantors": "A101",
    "property": "A124",
    "installment_plans": "A143",
    "housing": "A153",
    "job": "A173",
    "number_people_liable": 1,
    "telephone": "A191",
    "foreign_worker": "A201",
    "gender": "male",
    "label": 0,
    "predicted_score": 1
}
```

## Metrics Jobs

Model code includes a metrics function used to compute Group and Bias metrics.
The metrics function expects a DataFrame with at lease the following 4 columns: `score` (predictions), `label_value` (ground truths), `gender`, and `age_over_forty` (protected classes).

### Sample Inputs

Choose **one** of
 - `df_baseline_scored.json`
 - `df_sample_scored.json`
