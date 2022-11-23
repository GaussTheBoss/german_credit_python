import pickle
from typing import List

import pandas

from aequitas.preprocessing import preprocess_input_df
from aequitas.group import Group
from aequitas.bias import Bias


def init() -> None:
    """
    A function to load the trained model artifact (.pickle) as a glocal variable.
    The model will be used by other functions to produce predictions.
    """

    global logreg_classifier

    # load pickled logistic regression model
    logreg_classifier = pickle.load(open("logreg_classifier.pickle", "rb"))


def score(data: dict) -> dict:
    """
    A function to predict loan default/pay-off, given a loan application sample (record).

    Args:
        data (dict): input dictionary to be scored, containing predictive features.

    Returns:
        (dict): Scored (predicted) input data.
    """

    # Turn input data into a 1-record DataFrame
    data = pandas.DataFrame([data])

    # There are only two unique values in data.number_people_liable.
    # Treat it as a categorical feature, to mimic training process
    data.number_people_liable = data.number_people_liable.astype("category")

    # Alternitavely, these features can be saved (pickled) and re-loaded
    predictive_features = [
        "installment_plans",
        "job",
        "number_people_liable",
        "savings_account",
        "debtors_guarantors",
        "housing",
        "credit_amount",
        "installment_rate",
        "credit_history",
        "foreign_worker",
        "number_existing_credits",
        "purpose",
        "telephone",
        "present_residence_since",
        "checking_status",
        "duration_months",
        "present_employment_since",
        "property",
    ]

    # Predict using saved model
    data["predicted_score"] = logreg_classifier.predict(data[predictive_features])

    return data.to_dict(orient="records")[0]


def metrics(data: pandas.DataFrame) -> List[dict]:
    """
    A function to compute Group and Bias metrics on scored and labeled data, containing protected classes.

    Args:
        data (pandas.DataFrame): Dataframe of loan applications, including ground truths, predictions.

    Returns:
        (List[dict]): Group and Bias metrics for each protected class.
    """

    # To measure Bias towards gender, filter DataFrame to "score", "label_value" (ground truth), and
    # "gender" (protected attribute)
    data_scored = data[["score", "label_value", "gender", "age_over_forty"]]

    # Process DataFrame
    data_scored_processed, _ = preprocess_input_df(data_scored)

    # Group Metrics
    xtab, _ = Group().get_crosstabs(data_scored_processed)

    # Absolute metrics, such as 'tpr', 'tnr','precision', etc.
    absolute_metrics = Group().list_absolute_metrics(xtab)

    # DataFrame of calculated absolute metrics for each sample population group
    absolute_metrics_df = xtab[
        ["attribute_name", "attribute_value"] + absolute_metrics
    ].round(2)

    # For example:
    """
        attribute_name  attribute_value     tpr     tnr  ... precision
    0   gender          female              0.60    0.88 ... 0.75
    1   gender          male                0.49    0.90 ... 0.64
    2   age_over_forty  True                0.54    0.45 ... 0.23
    3   age_over_forty  False               0.45    0.54 ... 0.32
    """

    # Bias Metrics
    # Disparities calculated in relation gender for "male" and "female"
    bias_df = Bias().get_disparity_predefined_groups(
        xtab,
        original_df=data_scored_processed,
        ref_groups_dict={"gender": "male", "age_over_forty": "False"},
        alpha=0.05,
        mask_significance=True,
    )

    # Disparity metrics added to bias DataFrame
    calculated_disparities = Bias().list_disparities(bias_df)

    disparity_metrics_df = bias_df[
        ["attribute_name", "attribute_value"] + calculated_disparities
    ].round(3)

    # For example:
    """
        attribute_name	attribute_value    ppr_disparity   precision_disparity
    0   gender          female             0.714            1.417
    1   gender          male               1.000            1.000
    2   age_over_forty  True                0.54            1.234
    3   age_over_forty  False              1.000            1.000
    """

    # Output a JSON object of calculated metrics
    return {
        "group_metrics": absolute_metrics_df.to_dict(orient="records"),
        "bias_metrics": disparity_metrics_df.to_dict(orient="records"),
    }


# Test Script
if __name__ == "__main__":
    # Load model
    init()

    # Test scoring/inferences
    score_sample = pandas.read_json(
        "df_sample.json", orient="records", lines=True
    ).iloc[0]
    print(pandas.DataFrame([score(score_sample)]))
    print()

    # Test batch metrics
    metrics_sample = pandas.read_json(
        "df_sample_scored.json", orient="records", lines=True
    )
    bias = metrics(metrics_sample)
    print(pandas.DataFrame(bias["group_metrics"]))
    print()
    print(pandas.DataFrame(bias["bias_metrics"]))
