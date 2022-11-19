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
from fuzzywuzzy import fuzz

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

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

#model = KeyBERT()


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

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
#labels = ["Information", "Complain", "Appreciation"]
#template = "The sentiment of this review is {}"

for i in range(len(df['Comments'])):
    df['clean'][i] = clean_text(df['Comments'][i])
    #print(extractor(df['clean'][i]))
    #predictions = classifier(df['clean'][i], 
    #       labels)
    #pprint.pprint(predictions)

keywords = []
text1 = list(df['clean'])
for i in range(0,len(text1)-31,30):
    
    text = " ".join(text1[i:i+30])
    summarized =summarizer(text)
    print(summarized)
    keyword = list(extractor(summarized[0].get('summary_text')))
    print(keyword)
    keywords = keywords + keyword
    print(keywords)

def get_comments_keyword_df(keywords_list,df):
    dictionary = {
        "keyword":[],
        "comments": []
    }
    for word in keywords:
        comments = []
        for comment in df['Comments']:
            if word in str(comment):
                comments.append(comment)
                 
        dictionary["keyword"].append(word)
        dictionary["comments"].append(comments)
        dataframe = pd.DataFrame(data=dictionary)
        dataframe = dataframe.explode('comments')
        
    return dataframe

def get_most_relevant_keywords(dataframe, occurrence=10):
    grouped_df = df.groupby('keyword').count()
    most_used_keywords_df =  grouped_df[grouped_df['comments'] > most_used_range]
    cleaned_list = [element.lower() for element in most_used_keywords_df.index]
    return list(set(cleaned_list))


def remove_similar_words(words_list):
    same_words = []
    for i in words_list:
        for j in words_list:
            ratio  = fuzz.ratio(i,j)
            if ratio < 100 and ratio > 70:
                same_words.append(i)
                print(i,j,ratio)
    return list(set(same_words))

