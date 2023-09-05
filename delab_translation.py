################################################
# Python function to translate text            #
################################################

######################### packages
import os
from transformers import pipeline

######################### model
# model path
dirname = os.getcwd()
parentDirectory = os.path.dirname(dirname)
# parentDirectory = os.path.dirname(parentDirectory)
# modelPath_translation_de = os.path.join(parentDirectory, "models/opus-mt-de-en")
# modelPath_translation_en = os.path.join(parentDirectory, "models/opus-mt-en-de")

modelPath_translation_en = "Helsinki-NLP/opus-mt-de-en"
modelPath_translation_de = "Helsinki-NLP/opus-mt-en-de"

######################### function DE
de_en_pipeline = pipeline("translation", modelPath_translation_de)


def de_en_translation(text):
    translated_text = de_en_pipeline(text)
    return translated_text


######################### function EN
en_de_pipeline = pipeline("translation", modelPath_translation_en)


def en_de_translation(text):
    translated_text = en_de_pipeline(text)
    return translated_text
