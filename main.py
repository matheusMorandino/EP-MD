from geniusLyricsLib import *
import pandas as pd 
from tqdm import tqdm

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

print(">>> Carregando dados de entrada")
try:
    df = pd.read_csv("raw.csv")
except:
    print(">>> Dados não encontrados, recriando discografia")
    #Extracting the information of the N most popular songs of Metallica
    df0 = build_discography_data(['Metallica', 'Megadeth', 'AC/DC', 'Guns N Roses', 'Iron Maiden', \
                                'Red Hot Chili Peppers', 'Nirvana', 'Aerosmith', 'The Rolling Stones', \
                                'Queen', 'U2', 'Judas Priest', 'Bon Jovi', 'Kiss', 'Pink Floyd', 'Dire straits', \
                                'Ramones', 'Green Day', 'The cure', 'Blondie'],1000,access_token)

####################################
############ STEP 2) cleaning and transforming the data using functions created on helpers script
####################################

#Filter data to use songs that have lyrics.
df = df[df['lyric'].notnull()]
df['lyric'] = df['lyric'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))

df.to_csv('raw_new.csv',index=False)

print(">>> Adicionando decadas")
df = create_decades(df)

####################################
############ STEP 3) Stores unique words of each lyrics song into a new column called words 
####################################

#list used to store the words
words = []
#iterate trought each lyric and split unique words appending the result into the words list
df = df.reset_index(drop=True)
print(">>> Transformando letras em um BoW")
for word in tqdm(df['lyric'].tolist()):
    words.append(unique(lyrics_to_words(word).split()))

#create the new column with the information of words lists 
df['words'] = words

print(df.head())
print(df.columns)

####################################
############ STEP 4) Sentiment Analysis using VADER Sentiment Intensinty Model 
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

