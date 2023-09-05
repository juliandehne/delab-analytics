################################################
# Python function to classify tweets           #
################################################
# source: https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment

# packages
import os

import GPUtil
import torch
from scipy.special import softmax
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import AutoTokenizer


def get_available_memory():
    GPUs = GPUtil.getGPUs()
    return GPUs[0].memoryFree if GPUs else 0


def sentiment_scores(texts):
    torch.cuda.empty_cache()
    # model path
    dirname = os.getcwd()
    parentDirectory = os.path.dirname(dirname)
    # modelPath_sentiment = os.path.join(parentDirectory, "models/twitter-xlm-roberta-base-sentiment/")
    modelPath_sentiment = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    # tokenizer and model
    tokenizer_sentiment = AutoTokenizer.from_pretrained(modelPath_sentiment)
    config = AutoConfig.from_pretrained(modelPath_sentiment)
    model_sentiment = AutoModelForSequenceClassification.from_pretrained(modelPath_sentiment, from_tf=True)

    batch_size = 2
    results = []

    # Process texts in batches
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i: i + batch_size]

        # Tokenize the batch of texts
        encoded_input = tokenizer_sentiment(batch_texts, return_tensors='pt', padding=True, truncation=True,
                                            max_length=512)
        # encoded_texts = {key: tensor.to(device) for key, tensor in encoded_input.items()}

        # Get model predictions for the batch
        output = model_sentiment(**encoded_input)
        scores = output.logits.detach().numpy()

        # Apply softmax on scores
        scores = softmax(scores, axis=1)

        # Create list of dictionaries with scores for each text
        labels = ['sent_negative', 'sent_neutral', 'sent_positive']
        batch_results = [{labels[i]: score[i] for i in range(len(labels))} for score in scores]
        results.extend(batch_results)

    return results
