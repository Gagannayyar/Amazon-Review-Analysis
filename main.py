
from services.nltkfunctions import *
import pandas as pd
import datetime
from services.scraping import ExtractReviews as er
from random import choice
import os

#Taking User Input for link
url = str(input("Enter the product url: "))

#One time downloads
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

#Getting the proxy listcd
working_directory = os.getcwd()
print(working_directory)
wp = pd.read_csv('TravelSentiments/services/Working_Proxies.csv')
proxy_list = list(wp['http'])

#Soup Object
proxy= {'http': choice(proxy_list)}
soup = er.html_code(url,proxy)

#Getting Product Details
product_profile = {
"Name": str(er.get_product_name(soup)),
"Category": str(er.get_product_category(soup)),
"Total pages": str(er.get_total_pages_reviews(soup))
}
print(product_profile)


start_time = datetime.datetime.now()
print(f"Started at {start_time}")

#getting the dataframe

df = er.list_to_dataframe(er.get_reviews_body(url=url,pages=er.get_total_pages_reviews(soup)))
print("Step 1: Dataframe Uploaded")
print(df.head(20))


#Cleaning the text
df['CleanedText'] = df['comments'].apply(NltkTransformation.clean_text)
print("Step 3: Cleaned Text")
df.to_csv('reviews_data.csv')

#Sentnence Tokenization
print("Step 4: Tokenization Started")
df = NltkTransformation.sent_tokenize_service(df,column_name='CleanedText',
                                            new_column_name='CleanedText')
print("Step 4: Tokenization Ended")

#Opening the lists in the ddataframe
print("Step 5: Exploded Dataframe")
df = df.explode('CleanedText',ignore_index=True)
print(df['CleanedText'])

#Lemmentization and word tokenization
print("Step 6: Lemme Started")
df['lemmentzed'] = df['CleanedText'].apply(NltkTransformation.lemma_words_pos_filtered)
print("Step 6: Lemme Ended")

#Getting bigrams list
print("Step 7: Making bigram list started")
biagram_tuple_list = NltkTransformation.get_bigram_list(df,
                                                        column_name='lemmentzed',
                                                        window_size=5,
                                                        freq_filter=30)
print("Step 7: Making bigram list Ended")
unique_bigrams = NltkTransformation.unique_tuples(biagram_tuple_list)
print("Step 8: Unique Tuple ended")

#Getting bigrams list in the dataframe

print("Step 9: End bigram list started")
df['bigram_list'] = df['lemmentzed'].apply(NltkTransformation.findbigramsintext, list_of_tuples= unique_bigrams)
df['bigrams']=df['bigram_list'].apply(NltkTransformation.keepnonempty)
df['bigrams']=df['bigrams'].apply(NltkTransformation.flatten_list)
print("Step 10: 2nd list ended")

#Calculating the polarity
df = NltkTransformation.calculate_sentiments(df,column_name='CleanedText')
print("Step 11: Sentiments calculated")

now = df[['bigrams','polarity']].groupby('bigrams')['polarity'].mean().sort_values().head(20)

print(now)
now.to_csv(r'/home/heptagon/Downloads/archive/results.csv')
print(f"the type of now is {type(now)}")
end_time = datetime.datetime.now()
print(f"Completed at {end_time}")
print(f"Total time taken {end_time - start_time}")

