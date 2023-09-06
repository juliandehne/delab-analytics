################################################
# Python function to translate text            #
################################################

######################### packages
import os

import torch
from transformers import pipeline

######################### model
# model path
dirname = os.getcwd()
parentDirectory = os.path.dirname(dirname)

modelPath_translation_en = "Helsinki-NLP/opus-mt-de-en"
modelPath_translation_de = "Helsinki-NLP/opus-mt-en-de"


def translations(agg_df):
    en_de_pipeline = pipeline("translation", modelPath_translation_de)
    de_en_pipeline = pipeline("translation", modelPath_translation_en)

    def en_de_translation(text):
        translated_text = en_de_pipeline(text)
        return translated_text[0]["translation_text"]

    def de_en_translation(text):
        translated_text = de_en_pipeline(text)
        return translated_text[0]["translation_text"]

    agg_df["text_de"] = agg_df["text"].apply(lambda x: en_de_translation(x))
    agg_df["text_en"] = agg_df["text"].apply(lambda x: de_en_translation(x))
    del de_en_translation
    del en_de_translation
    torch.cuda.empty_cache()
    return agg_df
