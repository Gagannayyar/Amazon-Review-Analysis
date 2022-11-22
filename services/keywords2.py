import pandas as pd

df = pd.read_excel('Comments.xlsx')

import spacy
from spacy import displacy

NER = spacy.load("en_core_web_sm")
import re

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


df['clean'] = None


for i in range(len(df['Comments'])):
    df['clean'][i] = clean_text(df['Comments'][i])

text = []
dep = []
tag = []
pos = []
for i in df['clean']:
    if isinstance(i,str):
        print(i)
        doc = NER(i)
        for token in doc:
            text.append(token.text)
            dep.append(token.dep_)
            tag.append(token.tag_)
            pos.append(token.pos_)
