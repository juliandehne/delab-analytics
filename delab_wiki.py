import os
import pickle

import pandas as pd
import wikipediaapi
import spacy_udpipe
import inflect


def wiki_search(query):
    """
    Searches Wikipedia for a given query and returns the content if found.
    If there's an error or the page doesn't exist, it returns None.

    Parameters:
    - query (str): The search query.

    Returns:
    - str: The content of the Wikipedia page or None if not found or there's an error.
    """
    wiki_lang = wikipediaapi.Wikipedia('en')  # Using the English Wikipedia

    try:
        page = wiki_lang.page(query)
        if not page.exists():
            return None
        return page.text
    except Exception as e:
        # If there's any error, return None
        return None


def wiki_get(title):
    """
    Fetches a Wikipedia article for a given title and returns a dictionary with the title and its content.
    If there's an error or the page doesn't exist, it returns None.

    Parameters:
    - title (str): The title or subject of the Wikipedia article.

    Returns:
    - dict: A dictionary with the article title and its content, or None if not found or there's an error.
    """
    wiki_lang = wikipediaapi.Wikipedia(language='en', user_agent="delab/1.0")  # Using the English Wikipedia

    try:
        page = wiki_lang.page(title)
        if not page.exists():
            return None
        else:
            # Check if page belongs to the category "Disambiguation pages"
            for category in page.categories.values():
                if "disambiguation" in category.lower():
                    return None
            return {"noun": title, "article": page.text}
    except Exception as e:
        # If there's any error, return None
        return None


def delab_wikipedia(df):
    # Initialize UDPIPE model
    spacy_udpipe.download("en")  # for English; replace 'en' with other language codes if needed
    nlp = spacy_udpipe.load("en")

    def process_text(text):
        doc = nlp(text)
        processed = [(token.lemma_, token.pos_) for token in doc]
        return processed

    # Processing text to obtain upos and lemma
    processed_data = df['text_en'].apply(process_text)
    df['lemmata'] = processed_data.apply(lambda x: [elem[0] for elem in x])
    df['upos'] = processed_data.apply(lambda x: [elem[1] for elem in x])

    # Extract nouns
    # df_nouns = df[df['upos'].str.contains("NOUN")]
    df_nouns = df[df['upos'].apply(lambda x: 'NOUN' in x)]

    # Function to filter lemmas based on upos
    def filter_lemmas(row):
        return [lemma for upos, lemma in zip(row['upos'], row['lemmata']) if upos == 'NOUN']

    df_nouns['lemma'] = df_nouns.apply(filter_lemmas, axis=1)

    # Vector of unique nouns
    nouns = set(df_nouns['lemma'].explode())

    nouns = [word for word in nouns if not word.startswith(('#', '@'))]

    p = inflect.engine()
    singular_words = [p.singular_noun(word) if p.singular_noun(word) else word for word in nouns]

    nouns = set(nouns + singular_words)

    # Check local files
    nouns_to_scrape = nouns
    if os.path.exists("wikipedia/wiki_nouns.pkl"):
        with open("wikipedia/wiki_nouns.pkl", "rb") as file:
            nouns_scraped = pickle.load(file)
        nouns_to_scrape = list(set(nouns) - set(nouns_scraped))

    if nouns_to_scrape:
        wiki_content = [wiki_get(noun) for noun in nouns_to_scrape]
        wiki_content = [x for x in wiki_content if x is not None]

        # Transform into DataFrame
        wiki_content_df = pd.DataFrame(wiki_content)

        # If DataFrame is not empty, continue with processing
        if not wiki_content_df.empty:
            # Cleaning and sentence splitting
            wiki_content_df['article'] = wiki_content_df['article'].str.replace(r'([[:punct:]])([A-ZÄÖÜ])', r'\1 \2')
            wiki_content_df = wiki_content_df.explode('article', ignore_index=True)
            wiki_content_df['no_of_words'] = wiki_content_df['sentence'].str.split().str.len()

            # Keep sentences with a length of at least 10 words
            wiki_content_df = wiki_content_df[wiki_content_df['no_of_words'] >= 10]
            wiki_content_df.drop(columns=["no_of_words", "delete"], inplace=True, errors='ignore')

            # Write to local files
            if not os.path.exists("wikipedia"):
                os.makedirs("wikipedia")

            if os.path.exists("wikipedia/wiki_nouns.pkl"):
                with open("wikipedia/wiki_nouns.pkl", "rb") as file:
                    old_nouns = pickle.load(file)
                with open("wikipedia/wiki_data.pkl", "rb") as file:
                    old_data = pickle.load(file)

                all_nouns = old_nouns + nouns_to_scrape
                all_data = pd.concat([old_data, wiki_content_df], ignore_index=True)
            else:
                all_nouns = nouns_to_scrape
                all_data = wiki_content_df

            with open("wikipedia/wiki_nouns.pkl", "wb") as file:
                pickle.dump(all_nouns, file)
            with open("wikipedia/wiki_data.pkl", "wb") as file:
                pickle.dump(all_data, file)

    # Get Wikipedia data (only if the file exists)
    if os.path.exists("wikipedia/wiki_data.pkl"):
        with open("wikipedia/wiki_data.pkl", "rb") as file:
            wiki_data = pickle.load(file)
        wiki_content = wiki_data[wiki_data['noun'].isin(nouns)]
    else:
        wiki_content = pd.DataFrame()  # Empty DataFrame

    return wiki_content
