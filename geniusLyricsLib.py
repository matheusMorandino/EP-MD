import lyricsgenius as genius
import pandas as pd
import numpy as np
import string 

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('wordnet')
from pprint import pprint
from requests.exceptions import Timeout

from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def build_discography_data(artists, n, access_token):
    """
    This function builds a dataframe composed of the combined results of the 'search_data'
    given a list of artists
    parameters:
    artists = list containing the names of th bands/artists to be searched
    n = max numbers of songs
    access_token = your access token of the genius api
    """

    compound_df = pd.DataFrame(columns=['artist', 'title', 'album', 'date', 'lyric'])

    for band in artists:
        compound_df = compound_df.append(search_data(band, n, access_token), ignore_index=True)

    return compound_df

def search_data(query,n,access_token):
    """
    This function uses the library lyricsgenius to extract the fields
    title, artist, album, date and lyrics and stores them into a pandas dataframe
    parameters:
    query = artist or band to search
    n = max numbers of songs
    access_token = your access token of the genius api
    """
    
    api = genius.Genius(access_token)

    list_lyrics = []
    list_title = []
    list_artist = []
    list_album = []
    list_year = []

    while True:
        try:
            artist = api.search_artist(query,max_songs=n,sort='popularity')
            break
        except:
            print("Timeout during collection, retrying...")
            pass

    songs = artist.songs
    #pprint(vars(songs[0])+'\n-\n')
    for song in songs:
        list_lyrics.append(song.lyrics)
        list_title.append(song.title)
        list_artist.append(song.artist)
        for key, value in song._body.items():
            if key == 'release_date': # get song year  (because of AttributeError)
                try: 
                    list_year.append(value[:4])
                except:
                    list_year.append(9999)
            if key == 'album':        # get album name (because of AttributeError)
                try:
                    for key2, value2 in value.items():
                        if key2 == 'name':
                            list_album.append(value2)
                except:
                    list_album.append('NONE')
        #list_album.append(song.album) #AttributeError: 'Song' object has no attribute 'album' 11/11/2021
        #list_year.append(song.year)   #AttributeError: 'Song' object has no attribute 'year'  12/11/2021

    df = pd.DataFrame({'artist':list_artist,'title':list_title,'album':list_album, 
                       'date':list_year,
                       'lyric':list_lyrics})
    
    return df

def clean_lyrics(df,column):
    """
    This function cleans the words without importance and fix the format of the  dataframe's column lyrics 
    parameters:
    df = dataframe
    column = name of the column to clean
    """
    df = df
    df[column] = df[column].str.lower()
    df[column] = df[column].str.replace(r"verse |[1|2|3]|chorus|bridge|outro","").str.replace("[","").str.replace("]","")
    df[column] = df[column].str.lower().str.replace(r"instrumental|intro|guitar|solo","")
    df[column] = df[column].str.replace("\n"," ").str.replace(r"[^\w\d'\s]+","").str.replace("efil ym fo flah","")
    df[column] = df[column].str.strip()

    return df

def lyrics_to_words(document):
    """
    This function splits the text of lyrics to  single words, removing stopwords and doing the lemmatization to each word
    parameters:
    document: text to split to single words
    """
    stop_words = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stopwordremoval = " ".join([i for i in document.lower().split() if i not in stop_words])
    punctuationremoval = ''.join(ch for ch in stopwordremoval if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punctuationremoval.split())
    return normalized

def create_decades(df):
    """
    This function creates a new column called decades used to group the songs and lyrics by decade based on the date released 
    for each song
    parameters:
    df = dataframe
    """
    years = []
    decades = []
    df['date'].fillna(0)
    df['date'] = df['date'].astype("str")
    for i in df.index:
        years.append(df['date'].str.split("-")[i][0])
    df['year'] = years
    df['year'] = df['year'].astype("int")

    for year in df['year']:
        if year < 1960:
            decades.append("50s")
        if 1960 <= year < 1970:
            decades.append("60s")
        if 1970 <= year < 1980:
            decades.append("70s")
        if 1980 <= year < 1990:
            decades.append("80s")
        if 1990 <= year < 2000:
            decades.append("90s")
        if 2000 <= year < 2010:
            decades.append("00s")
        if 2010 <= year < 2020:
            decades.append("10s")
        if 2020 <= year :
            decades.append("20s")
    df['decade'] = decades
    df = df[['artist','title','album','decade','year','date','lyric']]
    return df



####################################
############ used in STEP 2
####################################

def unique(list1): 
     # intilize a null list 
    unique_list = [] 
      
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    
    return unique_list



####################################
############ used in STEP 4
####################################

def plot_wordcloud(df,row,col):
    wc = WordCloud( background_color="white",colormap="Dark2",max_font_size=100,random_state=15)

    fig = plt.figure(figsize=(20,10))
    
    for index, value in enumerate(df.columns[1:]):

        top_dict = dict(zip(df['words'].tolist(),df[value].tolist()))
        wc.generate_from_frequencies(top_dict)
        plt.subplot(row,col,index+1)
        plt.imshow(wc,interpolation="bilinear")
        plt.axis("off")
        plt.title(f"{value}",fontsize=15) 

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def words_stats(df,main_df):
    unique_words = []
    total_words = []
    total_news = []
    years = []
    for value in df.columns[1:]:
         unique_words.append(np.count_nonzero(df[value]))
         total_words.append(sum(df[value]))
         years.append(str(value))
         total_news.append(main_df['decade'][main_df['decade']==value].count())
    
    data = pd.DataFrame({'decade':years,'unique words':unique_words,'total words':total_words,'total songs':total_news})
    data['words per songs'] = round(data['total words'] / data['total songs'],0)
    data['words per songs'] = data['words per songs'].astype('int')
    return data

def plot_freq_words(df,decade,n_words):
    
    top_words_2020 = df.sort_values([decade],ascending=False).head(n_words)

    fig = plt.figure(figsize=(15,8))
    plt.plot(top_words_2020['words'],top_words_2020[df.columns[1]])
    plt.plot(top_words_2020['words'],top_words_2020[df.columns[2]])
    plt.plot(top_words_2020['words'],top_words_2020[df.columns[3]])
    plt.plot(top_words_2020['words'],top_words_2020[df.columns[4]])

    plt.legend(df.columns[1:].tolist())
    plt.title(f"Most frequent words in {decade} compared with other decades",fontsize=14)
    plt.xlabel(f'Most Frequent Words of {decade}',fontsize=12)
    plt.ylabel('Frecuency',fontsize=12)
    plt.xticks(fontsize=12,rotation=20)
    plt.yticks(fontsize=12)
    plt.savefig("lineplot.png",format="png", bbox_inches='tight')
    plt.show()


#Function that display the most common words used per decade
def unique_decade_words(df,decade,n_words):
    periods = []
    unique_words = []
    top_words = df.sort_values([decade],ascending=False).head(n_words).reset_index(drop=True)

    for d in top_words.columns[1:]:
        if d != decade:
            periods.append(d)

    for i in top_words.index:
        if top_words[periods[0]].iloc[i]<top_words[decade].min():
            if top_words[periods[1]].iloc[i]<top_words[decade].min():
                if top_words[periods[2]].iloc[i]<top_words[decade].min():
                    unique_words.append(top_words['words'].iloc[i])
    return unique_words