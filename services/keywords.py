from keybert import KeyBERT
import pandas as pd
import re
from transformers import pipeline
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np

# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])

model_name = "ml6team/keyphrase-extraction-kbir-inspec"
extractor = KeyphraseExtractionPipeline(model=model_name)


#model = KeyBERT()


#summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

#keywords = model.extract_keywords(text, keyphrase_ngram_range=(3, 3), 
#                               stop_words='english',
#                              use_maxsum=True, nr_candidates=20, top_n=5)

def clean_text(text) -> str:
        """
        Purpose: Remove punctuations etc. using regex
        Params:
            text: A sentence/collection of strings
        """
        text=re.sub(r"\?", ".", str(text))
        text=re.sub(r"\!", ".", text)
        text=re.sub(r'([.])\1+', r'\1', text)
        rexp=r"\.(?=[A-Za-z]{1,15})"
        text=re.sub(rexp, ". ", text)
        return text
df = pd.read_excel('Comments.xlsx')
df['clean'] = None
for i in range(len(df['Comments'])):
    df['clean'][i] = clean_text(df['Comments'][i])

text1 = list(df['clean'])
for i in range(0,len(text1),60):
    text = " ".join(text1[i:i+60])
    #print(summarizer(text))
    print(extractor(text))
