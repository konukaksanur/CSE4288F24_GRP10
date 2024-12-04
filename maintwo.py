import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
import re
import nltk         #pip install nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem  import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from dataPreprocessor import DataPreprocessing

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger_eng')

def readDataset(FileName ):
    column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(FileName, encoding='latin1', names=column_names)
    df = df.head(10)
    print("\n" , df.head())
    print("\n" , df.tail() , "\n")
    print("Dataset is read.")
    return df

def data_details(df):
    print("\n" , df.info())
    print("\nMissing value count \n")
    print(df.isnull().sum())
    print("\nDuplicated value count" , df.duplicated().sum())
    return df

def get_wordnet_pos(word):
        """Get the part of speech tag for a word."""
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

def data_preprocessing(df):
    print("***********")
    print(df)
    df_filtered = df[['target' , 'text']]
    print(df_filtered)

    df_filtered.loc[:, 'text'] = df_filtered['text'].apply(lambda x: x.lower())
    df_filtered.loc[:, 'text'] = df_filtered['text'].replace(r'@[A-Za-z0-9_]+', '', regex=True)
    df_filtered.loc[:, 'text'] = df_filtered['text'].replace(r'http\S+', '', regex=True)
    df_filtered.loc[:, 'text'] = df_filtered['text'].replace(r'www\S+', '', regex=True)
    df_filtered.loc[:, 'text'] = df_filtered['text'].replace(r'[0-9]+', '', regex=True)

    #bunu da daha optimize yazabiliriz x,y,z kısmı değiştirilecek,yerinegelen,silinecekse yukardakileri koyabiliriz mesela silinecek yere ve '' atar yerine?
    #hayır maketrans sadece tekli karakterlerde çalışıyor..!
    table = str.maketrans('', '', string.punctuation)
    texts = df_filtered.loc[:,'text'].copy()
    new_text = [' '.join([w.translate(table) for w in text.split()]) for text in texts]
    df_filtered.loc[:,'text'] = new_text

    # bu kısım words diye sütun oluşturup hepsini dizi gibi virgüllü yazıyo cümlelerin
    df_filtered.loc[:, 'words'] = df_filtered['text'].apply(word_tokenize)
    # sürekli aynı sütunları ekliyo gibi olduk azalta azalta hımhım 
    #text --- words---- wordscleaned bunlar hep azaltıla azaltıla hepsi tutuluyo içinde

    #ing kelimelerin olduğu byüüüükcene bi sözlük gibi düsün icinde olmayanları silcez
    stop_words = set(stopwords.words('english'))

    #burada tekrar lower yapmamız çok saçma yukarıda yapıyoruz zaten tekrar olmuş
    df_filtered.loc[:, 'words_cleaned'] = df_filtered['words'].apply(lambda words: [word for word in words if word.lower() not in stop_words])
    print(df_filtered[['text' , 'words', 'words_cleaned']].head())

    
    #bu arada tak diye yeni sütun eklemesi pandastan geliyormus wo
     # Lemmatization
     #eğer türünü vermezsen kelimenin varsayılan olarak noun cevirir ki bu da runningi tekrar running olarak çevirir
    lemmatizer = WordNetLemmatizer()
    df_filtered['words_lemmatized'] = df_filtered['words_cleaned'].apply(
        lambda words: [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
    )

    # Join the lemmatized words back into a string
    df_filtered['lemmatized_text'] = df_filtered['words_lemmatized'].apply(lambda words: ' '.join(words))


    return df_filtered    

    
    

#dün iremin söylediği mantık sadce fonka topladım 
def pipelineFunc():
    df = readDataset("training.1600000.processed.noemoticon.csv")
    df = data_details(df)
    df= data_preprocessing(df)
    print(df[['words_cleaned' , 'words_lemmatized']].head(10))





pipelineFunc()


