# #######################
# Python function to translate text            #
# #######################


import torch
#  packages
from tqdm import tqdm
from transformers import TranslationPipeline, AutoModelForSeq2SeqLM, AutoTokenizer

#  model
# model path
modelPath_translation_en = "Helsinki-NLP/opus-mt-de-en"
modelPath_translation_de = "Helsinki-NLP/opus-mt-en-de"


def translations(agg_df, batch_size=32):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load model and tokenizer for 'en-de' translation
    # Load model and tokenizer for 'en-de' translation and move to GPU
    model_en_de = AutoModelForSeq2SeqLM.from_pretrained(modelPath_translation_de).to(device)
    tokenizer_en_de = AutoTokenizer.from_pretrained(modelPath_translation_de)
    en_de_pipeline = TranslationPipeline(model=model_en_de, tokenizer=tokenizer_en_de,
                                         device=0 if device == "cuda" else -1)

    # Load model and tokenizer for 'de-en' translation and move to GPU
    model_de_en = AutoModelForSeq2SeqLM.from_pretrained(modelPath_translation_en).to(device)
    tokenizer_de_en = AutoTokenizer.from_pretrained(modelPath_translation_en)
    de_en_pipeline = TranslationPipeline(model=model_de_en, tokenizer=tokenizer_de_en,
                                         device=0 if device == "cuda" else -1)

    # Translation function using batches
    def batch_translate(texts, translation_pipeline):
        translated_texts = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i: i + batch_size]
            translations = translation_pipeline(batch)
            translated_texts.extend([t['translation_text'] for t in translations])
        return translated_texts

    # Split the DataFrame's 'text' column into batches and translate them
    agg_df["text_de"] = batch_translate(agg_df["text"].tolist(), en_de_pipeline)
    agg_df["text_en"] = batch_translate(agg_df["text"].tolist(), de_en_pipeline)

    # Cleanup
    del de_en_pipeline, en_de_pipeline
    torch.cuda.empty_cache()

    return agg_df
