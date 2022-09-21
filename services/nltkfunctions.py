"""
Purpose: The nltk functions required for the transformation and calculation of the sentiments
Created By: Gagan Nayyar
Contact: gagan@heptagon.in
Version: V.1.0.0
Created On: 13-07-2022
"""

#Importing Libraries
import pandas as pd
import re
import nltk
from nltk.collocations import *
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.collocations import *
from textblob import TextBlob
from pydantic import BaseModel


class NltkTransformation(BaseModel):

    def __init__() -> None:
        pass


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

    def sent_tokenize_service(df, column_name, new_column_name) -> pd.DataFrame:
        """Purpose:Sentence Tokenization
        Params:
            df: the dataframe containing the reviews
            column_name: The name of the column containing the reviews
            new_column_name: The name of the new column to be created
        """
        df[new_column_name] = df[column_name].apply(sent_tokenize)
        return df

    def lemma_words_pos_filtered(text) -> str:
        """
        Purpose: word tokenization and lemmatization
        Params:
            text: A sentence/collection of strings
        """
        word_list=[]
        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(text)
        pos_tokens=nltk.pos_tag(tokens)
        for pos in pos_tokens:
            if (pos[1].startswith('N')):
                word_list=word_list+[lemmatizer.lemmatize(pos[0].lower(), wordnet.NOUN)]
            if (pos[1].startswith('V')):
                word_list=word_list+[lemmatizer.lemmatize(pos[0].lower(), wordnet.VERB)]
            if (pos[1].startswith('J')):
                word_list=word_list+[lemmatizer.lemmatize(pos[0].lower(), wordnet.ADJ)]
        word_list=[word for word in word_list if word not in  
                                        stopwords.words('english') ]
        return  " ".join(word_list)

    def get_bigram_list(df,column_name,window_size: int,freq_filter: int) -> list:
        """
        Purpose: Getting the list of bigrams
        Params:
            df: The dataframe
            column_name: The name of the column which is lemmentized
            window_size: The number of windows to be sepcified for finder
            freq_finder: The number of the frequency of filters required
        """
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words( 
        " ".join(df[column_name]).split(),
        window_size=window_size)
        finder.apply_freq_filter(freq_filter)
        bigram_list_tuples=finder.nbest(
                     bigram_measures.likelihood_ratio, 30)
        


        return bigram_list_tuples

    def unique_tuples(list_of_tuples) -> list:
        """
        Purpose: getiing the unique list of bigrams
        Params:
            list_of_tuples: Getting the list of tuples which are unique
        """
        list_ = [tuple(sorted(t)) for t in list_of_tuples]
        list_ = list(set(list_))
        print("Step 5: LIst sorted and duplicates removed")
        return list_

    def findbigramsintext(text,list_of_tuples) -> list:
        mylist=nltk.word_tokenize(text)
        list1=[x for x in mylist]
        feature_list = []
        for i in range(len(list_of_tuples)):
            feature_list.append([])
            i=0
        for l in list_of_tuples:
            list2=[x for x in l]
            result =  all(elem in list1  for elem in list2)
            if result: 
                feature_list[i].append(' '.join(list2))
                i=i+1


        return feature_list

    def keepnonempty(list1) -> list:
        """
        Purpose: Keeping only the non empty list in the comments column of dataframe
        Param:
            list1: Bigram List
        """

        mylist= [x for x in list1 if x != []]
        return mylist

    def flatten_list(row_list) -> list:
        l = [item for inner_list in row_list for item in inner_list]
        return l

    def calculate_sentiments(df,column_name) -> pd.DataFrame:
        df['polarity'] = df[column_name].apply(lambda x: 
                                           TextBlob(x).sentiment[0])
        df = df.explode('bigrams', ignore_index=False)
        return df


