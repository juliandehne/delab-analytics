import pandas as pd

from delab_bertopic import delab_bertopic, init_topic_models
from delab_wiki import delab_wikipedia


def delab_topics(df, wikipedia="yes"):
    # Split dataframe by unique tree_id
    unique_tree_ids = df['tree_id'].unique()
    final_results = []

    # put this outside the loop because it takes a long time to initialize
    topic_model, vectorizer_model, representation_model = init_topic_models()

    for tree_id in unique_tree_ids:
        df_sub = df[df['tree_id'] == tree_id]

        # Aggregation
        df_agg = df_sub.groupby(['row_id', 'doc_id']).head(1)
        df_agg = df_agg[['text', 'row_id', 'conversation_id', 'id']]

        # Cleaning
        df_agg['text'] = df_agg['text'].str.replace(r"@.+?\b", "")
        http_link = r'(http|https)[^[\s"\<&\#\n\r]]+'
        df_agg['text'] = df_agg['text'].str.replace(http_link, "")

        # Enrich data with Wikipedia data if required
        if wikipedia == "yes" and len(df_agg) <= 1000:
            df_wikipedia = delab_wikipedia(df_sub)

            n_sample = 1000 - len(df_agg)
            n_sample = min(len(df_wikipedia), n_sample)

            df_wikipedia = df_wikipedia.sample(n=n_sample)
            df_wikipedia = df_wikipedia.rename(columns={"sentence": "text"})

            df_agg = pd.concat([df_agg, df_wikipedia], ignore_index=True, sort=False)

        # Run bertopic
        topics = delab_bertopic(df_agg, topic_model, vectorizer_model, representation_model)
        topics_prob = topics.get("topic_prob")

        # Merge
        df_agg = pd.merge(df_agg, topics_prob,
                          on=['row_id', 'conversation_id', 'id', 'text'],
                          how='left')

        # Cleaning
        df_agg = df_agg.dropna(subset=['row_id'])
        df_agg = df_agg.sort_values(by='row_id')

        final_results.append(df_agg)

    # Concatenate results from all tree_ids
    final_df = pd.concat(final_results, ignore_index=True)

    return final_df

# You'd call the function like this:
# result = delab_topics(df, wikipedia="yes")
