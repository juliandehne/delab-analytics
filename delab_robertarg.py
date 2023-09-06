################################################
# Training implicit arguments                  #
################################################

######################### packages
import gc
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def preprocess_twitter_text(text):
    """
    Preprocesses Twitter text by removing @mentions, URLs, RTs, and other undesired components.

    Parameters:
    - text (str): the raw Twitter text.

    Returns:
    - str: the preprocessed Twitter text.
    """

    # Remove RT (retweet marker)
    text = re.sub(r'\bRT\b', '', text)

    # Remove @mentions
    text = re.sub(r'@\w+', '', text)

    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # Remove special characters and numbers (optional, based on your requirements)
    text = re.sub(r'[^A-Za-z\s.,?!;:"\'-]', '', text)

    # Remove additional white spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def arg_prediction_batch(texts):
    texts = list(map(preprocess_twitter_text, texts))
    device, model_topics, tokenizer_topics = get_robert_arg_model()

    batch_size = 4

    # Split texts into chunks of batch_size
    batched_texts = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    all_results = []

    for batch in tqdm(batched_texts):
        # Tokenize texts in the current batch
        args = tokenizer_topics(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Send the tokenized batch to GPU
        for key in args:
            args[key] = args[key].to(device)

        # Predict
        logits = model_topics(**args).logits

        # Apply softmax to logits (and transfer to CPU if needed for further processing)
        results = torch.softmax(logits, dim=1).cpu().tolist()

        all_results.extend(results)

    del model_topics
    del tokenizer_topics

    torch.cuda.empty_cache()

    gc.collect()

    return all_results


def get_robert_arg_model():
    pathFinetuned_topics = "./models/custom_models/robertarg-fine"
    if os.path.exists(pathFinetuned_topics):
        modelPath_topics = pathFinetuned_topics
    else:
        modelPath_topics = "chkla/roberta-argument"
    tokenizerPath_topics = "chkla/roberta-argument"

    # Check CUDA availability
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    # Initialize tokenizer
    tokenizer_topics = AutoTokenizer.from_pretrained(tokenizerPath_topics)
    # Load model to GPU
    model_topics = AutoModelForSequenceClassification.from_pretrained(modelPath_topics).to(device)
    return device, model_topics, tokenizer_topics


def arg_predictions(texts):
    device, model_topics, tokenizer_topics = get_robert_arg_model()

    def arg_prediction(text):
        arg = tokenizer_topics(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        for key in arg:
            arg[key] = arg[key].to(device)
        arg_classification_logits = model_topics(**arg).logits
        arg_results = torch.softmax(arg_classification_logits, dim=1).cpu().tolist()[0]
        return arg_results

    texts = list(map(arg_prediction, texts))
    return texts
