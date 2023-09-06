import numpy as np
import pandas as pd
from langdetect import detect
from sentence_splitter import split_text_into_sentences

from delab_robertarg import arg_prediction_batch
from delab_sentiment import sentiment_scores
# Define the custom aggregation function
from delab_topics import delab_topics
from delab_translation import translations


def analyze(df: pd.DataFrame, scope="all"):
    assert "tree_id" in df.columns, "conversation_id needs to be named tree_id for grouping conversations"
    assert "text" in df.columns , "rows must contain a text column"

    limit_languages(df)

    # 1. Split the 'text' column into multiple rows
    # Split the 'text' column into multiple rows using the detected language
    # Create a new dataframe with expanded rows while retaining all original columns
    print("1. Splitting Sentences")
    sentences_df = split_sentences(df)

    # 2. Perform a couple of functions that are defined on sentences
    if scope == "argument_prediction":
        print("2.1 Argument mining")
        sentences_df = predict_argument(sentences_df, text_column="sentence")
    if scope == "sentiment":
        print("2.2 Sentiment analysis")
        sentences_df = predict_sentiment(sentences_df, text_column="sentence")
    if scope == "all":
        print("2.1 Argument mining")
        sentences_df = predict_argument(sentences_df, text_column="sentence")
        print("2.2 Sentiment analysis")
        sentences_df = predict_sentiment(sentences_df, text_column="sentence")

    # 3. Aggregate the results (For this example, let's say we concatenate the sentences back and sum the word counts)
    print("3. Aggregating Sentences")
    agg_df = sentences_df.groupby('sentence_group').agg(default_agg).reset_index(drop=True)

    done_translations = False
    if scope == "translations" or scope == "all":
        print("4.1 doing Translations")
        agg_df = translations(agg_df)
        done_translations = True

    # 4. perform functions that are defined on texts (probably topic detection for instance)
    if scope == "topics" or scope == "all":
        if not done_translations:
            print("4.1 doing Translations")
            agg_df = translations(agg_df)
        print("4.2 computing topics")
        agg_df = delab_topics(agg_df, wikipedia="yes")

    return agg_df


def split_sentences(df):
    rows = []
    for sentence_group, row in df.iterrows():
        sentences = split_text_into_sentences(row['text'], row['language'])
        for sentence in sentences:
            new_row = row.copy()
            new_row['sentence'] = sentence
            new_row['sentence_group'] = sentence_group
            rows.append(new_row)
    sentences_df = pd.DataFrame(rows).reset_index(drop=True)
    # Drop rows where 'text' column has NaN values
    sentences_df.dropna(subset=['sentence'], inplace=True)
    # Drop rows where 'text' column is an empty string or just whitespace
    sentences_df = sentences_df[sentences_df['sentence'].str.strip() != ""]
    return sentences_df


def limit_languages(df):
    # check language first and foremost
    if "language" not in df.columns:
        df['language'] = df['text'].apply(detect)
    # Replace any non-'en' or non-'de' languages with 'en' (we assume en or de for all)
    df['language'] = df['language'].apply(lambda lang: lang if lang in ['en', 'de'] else 'en')


def default_agg(x):
    """
        For numeric columns: Compute the mean.
        For non-numeric (text) columns:
            If all values are the same, it will return the first value.
            If there are multiple unique values, it will concatenate them.
    :param x:
    :return:
    """

    # If column is numeric
    if np.issubdtype(x.dtype, np.number):
        if x.nunique() == 1:
            return x.iloc[0]
        return x.mean()

    # If column is non-numeric
    else:
        unique_vals = x.unique()
        if len(unique_vals) == 1:
            return unique_vals[0]
        else:
            return ' '.join(map(str, unique_vals))


def predict_argument(df, text_column="text"):
    argument_predictions = arg_prediction_batch(list(df[text_column]))
    # argument_predictions = arg_predictions(list(df[text_column]))
    argument_predictions = pd.DataFrame(argument_predictions, columns=["p_is_argument", "p_is_not_argument"])
    df = df.join(argument_predictions)
    return df


def predict_sentiment(df, text_column="text"):
    texts = list(df[text_column])
    scores = sentiment_scores(texts)
    sentiment_data = pd.DataFrame.from_dict(scores)
    sentiment_data.reset_index(inplace=True, drop=True)
    df.reset_index(inplace=True, drop=True)
    result = df.join(sentiment_data)
    return result
