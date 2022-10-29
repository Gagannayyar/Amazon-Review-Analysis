import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

text = "Nick likes to play football, however he is not too fond of tennis."
text_tokens = word_tokenize(text)

tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]

print(tokens_without_sw)

df = pd.read_csv("reviews_data.csv")

def bigrams(doc):
    result = list()
    sentence = list()

    for token in doc:
        if token.is_alpha:
            sentence.append(token)

    for word in range(len(sentence)-1):
        first_word = sentence[word]
        second_word = sentence[word+1]
        element = [first_word, second_word]
        result.append(element)

    return result

def remove_stop_words(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    filtered_sentence = (" ").join(tokens_without_sw)
    return filtered_sentence

nlp = spacy.load("en_core_web_sm")

for sent in df['CleanedText']:
    if isinstance(sent, str):
        sent = remove_stop_words(sent)
        sent2 = nlp(sent)
        ngrams = bigrams(sent2)
        print(f"---ngrams:{ngrams}------")
        print(f"---------{sent}---------")
        for token in sent2:
            if token.dep_ == "nsubj" or token.dep_ == "amod" or token.dep_ == "compound":
                print(token.text,  token.dep_)
