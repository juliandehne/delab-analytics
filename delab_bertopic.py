################################################
# Python function to run BERTopic             #
################################################

######################### packages
import os
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.representation import TextGeneration

import nltk
from nltk.corpus import stopwords

from transformers import pipeline


######################### specify topic model
#model path
dirname = os.getcwd()
parentDirectory = os.path.dirname(dirname)
# modelPath = os.path.join(parentDirectory, "models/distiluse-base-multilingual-cased-v1/")
modelPath = "sentence-transformers/distiluse-base-multilingual-cased-v1"

#topic model
topic_model = BERTopic(embedding_model=modelPath,
                       calculate_probabilities=True,
                       language="multilingual",
                       nr_topics="auto",
                       min_topic_size=10,
                       low_memory=True,
                       verbose=True)

######################### specify flan text2text model
#model path
dirname = os.getcwd()
parentDirectory = os.path.dirname(dirname)
# modelPath = os.path.join(parentDirectory, "models/flan-t5-base/")
modelPath = "google/flan-t5-base"

#specify pipeline
prompt = "I have a topic described by the following keywords: [KEYWORDS]. Based on the previous keywords, what is this topic about?"
generator = pipeline('text2text-generation', model=modelPath)
representation_model = TextGeneration(generator, prompt)

######################### specify parameters to update topic representation
#vectorizer
#https://maartengr.github.io/BERTopic/getting_started/vectorizers/vectorizers.html#ngram_range
nltk.download('stopwords')
list_of_stopwords = stopwords.words('english')
stopwords_german = stopwords.words('german')
list_of_stopwords.extend(stopwords_german)
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=list_of_stopwords)


######################### tm function
def delab_bertopic(docs):

    ######################### extract text to train the model
    text = docs["text"].to_list()

    ######################### train model
    topics, probs = topic_model.fit_transform(text)

    ######################### update topics and representations
    #use Flan-T5 language model to summarize top words
    topic_model.update_topics(text,
                              vectorizer_model=vectorizer_model,
                              representation_model=representation_model)

    ######################### make nice(r) topic labels
    topic_labels = topic_model.generate_topic_labels(nr_words=1,
                                                     topic_prefix=False)

    #set topic labels
    #https://maartengr.github.io/BERTopic/getting_started/topicrepresentation/topicrepresentation.html#custom-labels
    topic_model.set_topic_labels(topic_labels)

    #make df of probabilities
    topic_prob = pd.concat([topic_model.get_document_info(text), pd.DataFrame(probs)], axis=1)
    topic_prob = pd.concat([topic_prob, pd.DataFrame(docs)], axis=1)

    #response
    response = dict()
    response["topic_prob"] = topic_prob
    response["topic_info"] = topic_model.get_topic_info()
    return response
