'''
pip install vaderSentiment
pip install lyricsgenius
pip install pandas
pip install wordcloud
pip install sklearn

RUN ON IDLE:
>>> dler = nltk.downloader.Downloader()
>>> dler._update_index()
>>> dler.download('wordnet')
>>> dler.download('vader_lexicon')
'''

#libraries used to extract, clean and manipulate the data
from geniusLyricsLib import *
import pandas as pd 
import numpy as np
import string

#To plot the graphs
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.style.use('seaborn')

#library used to count the frequency of words
from sklearn.feature_extraction.text import CountVectorizer

#To create the sentiment analysis model, tokenization and lemmatization
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize
import nltk.data
#nltk.download('vader_lexicon')
#nltk.download('punkt')



####################################
############ STEP 1) search data in genius
####################################

access_token = '5yWU82ZtyFHn9FBRC314WtNowapfuTFRwMho-82bR3gCmWFzwaekAV2oYqdHAsLm'

try:
    df = pd.read_csv("raw.csv")
except:
    #Extracting the information of the N most popular songs of Metallica
    df0 = build_discography_data(['Metallica', 'Megadeth', 'AC/DC', 'Guns N Roses', 'Iron Maiden', \
                                'Red Hot Chili Peppers', 'Nirvana', 'Aerosmith', 'The Rolling Stones', \
                                'Queen', 'U2', 'Judas Priest', 'Bon Jovi', 'Kiss', 'Pink Floyd', 'Dire straits', \
                                'Ramones', 'Green Day', 'The cure', 'Blondie'],1000,access_token)

    df = clean_lyrics(df0,'lyric')


####################################
############ STEP 2) cleaning and transforming the data using functions created on helpers script
####################################


df.to_csv('raw.csv',index=False)

df = create_decades(df)

#Filter  data to use songs that have lyrics.
df = df[df['lyric'].notnull()]

df.to_csv('lyrics.csv',index=False)



####################################
############ STEP 3) Stores unique words of each lyrics song into a new column called words 
####################################

#list used to store the words
words = []
#iterate trought each lyric and split unique words appending the result into the words list
df = df.reset_index(drop=True)
for word in df['lyric'].tolist():
    words.append(unique(lyrics_to_words(word).split()))

#create the new column with the information of words lists 
df['words'] = words



####################################
############ STEP 4) Create a new dataframe of all the words used in lyrics and its decades 
####################################

#list used to store the information
set_words = []
set_decades = []

#Iterate trought each word and decade and stores them into the new lists
for i in df.index:
    for word in df['words'].iloc[i]:
        set_words.append(word)
        set_decades.append(df['decade'].iloc[i])

#create the new data frame  with the information of words and decade lists 
words_df = pd.DataFrame({'words':set_words,'decade':set_decades})


#Defined  your own Stopwords in case the clean data function does not remove all of them
stop_words = ['verse','im','get','1000','58','60','80','youre','youve',
               'guitar','solo','instrumental','intro','pre',"3","yo","yeah"]

# count the frequency of each word that don't have on the stop_words lists          
cv = CountVectorizer(stop_words=stop_words)

#Create a dataframe called data_cv to store the the number of times the word was used in  a lyric based their decades
text_cv = cv.fit_transform(words_df['words'].iloc[:])
print(text_cv)
data_cv = pd.DataFrame(text_cv,columns=cv.get_feature_names())
data_cv['decade'] = words_df['decade']

#created a dataframe that Sums the ocurrence frequency of each word and group the result by decade
vect_words = data_cv.groupby('decade').sum().T
vect_words = vect_words.reset_index(level=0).rename(columns ={'index':'words'})
vect_words = vect_words.rename_axis(columns='')

#Save the data into a csv file
vect_words.to_csv('words.csv',index=False)

#change the order of columns to order from the oldest to actual decade
vect_words = vect_words[['words','80s','90s','00s','10s']]
vect_words = vect_words[['words','90s']] # Used for testing with a small N


#words_stats(vect_words,df)               
#plot_wordcloud(vect_words,2,2)           
#plot_freq_words(vect_words,'80s',10)     
#unique_decade_words(vect_words,'80s',10) 



####################################
############ STEP 5) Sentiment Analysis using VADER Sentiment Intensinty Model
####################################

#Create lists to store the different scores for each word
negative = []
neutral = []
positive = []
compound = []

#Initialize the model
sid = SentimentIntensityAnalyzer()

#Iterate for each row of lyrics and append the scores
for i in df.index:
    scores = sid.polarity_scores(df['lyric'].iloc[i])
    negative.append(scores['neg'])
    neutral.append(scores['neu'])
    positive.append(scores['pos'])
    compound.append(scores['compound'])

#Create 4 columns to the main data frame  for each score 
df['negative'] = negative
df['neutral'] = neutral
df['positive'] = positive
df['compound'] = compound

print(df.head())

#Save the data into a csv file
df.drop(columns=['date','words']).to_csv('lyrics_analysis.csv',index=False)

for name, group in df.groupby('decade'):
    plt.scatter(group['positive'],group['negative'],label=name)
    plt.legend(fontsize=10)

plt.xlim([-0.05,0.7])
plt.ylim([-0.05,0.7])

plt.title("Lyrics Sentiments by Decade")
plt.xlabel('Positive Valence')
plt.ylabel('Negative  Valence')
plt.show()                               ############################# UNCOMENT TO PLOT THE LYRICS SENTIMENTS BY DECADE
