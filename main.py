import pandas as pd
from delab_robertarg import arg_prediction_batch
from delab_sentiment import sentiment_scores


def predict_argument(df):
    argument_predictions = arg_prediction_batch(list(df.text))
    argument_predictions = pd.DataFrame(argument_predictions, columns=["p_is_argument", "p_is_not_argument"])
    df = df.join(argument_predictions)
    return df


def predict_sentiment(df):
    texts = list(df.text)
    scores = sentiment_scores(texts)
    sentiment_data = pd.DataFrame.from_dict(scores)
    sentiment_data.reset_index(inplace=True, drop=True)
    df.reset_index(inplace=True, drop=True)
    result = df.join(sentiment_data)
    return result


def analyze(df: pd.DataFrame, scope="all"):
    if scope == "argument_prediction":
        df = predict_argument(df)
    if scope == "sentiment":
        df = predict_sentiment(df)
    if scope == "all":
        df = predict_argument(df)
        df = predict_sentiment(df)
    return df
