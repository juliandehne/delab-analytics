################################################
# Training implicit arguments                  #
################################################

######################### packages
import gc
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def arg_prediction_batch(texts):
    ######################### tokenizer
    # tokenizer path
    dirname = os.getcwd()
    parentDirectory = os.path.dirname(dirname)
    tokenizerPath_topics = os.path.join(parentDirectory, "models/roberta-argument")
    if not os.path.exists(tokenizerPath_topics):
        tokenizerPath_topics = "chkla/roberta-argument"
        # raise FileNotFoundError(f"The trained model in '{tokenizerPath_topics}' does not exist.")

    ######################### model
    # model path
    pathFinetuned_topics = "./models/custom_models/robertarg-fine"

    if os.path.exists(pathFinetuned_topics):
        modelPath_topics = pathFinetuned_topics
    else:
        modelPath_topics = "chkla/roberta-argument"

    # Check CUDA availability
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Initialize tokenizer
    tokenizer_topics = AutoTokenizer.from_pretrained(tokenizerPath_topics)

    # Load model to GPU
    model_topics = AutoModelForSequenceClassification.from_pretrained(modelPath_topics).to(device)

    batch_size = 2

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
