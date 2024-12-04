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




def readDataset(FileName ):
    column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(FileName, encoding='latin1', names=column_names)
    print("\n" , df.head())
    print("\n" , df.tail() , "\n")
    
    print("Dataset is read.")
    return df

def data_details(df):
    print("\n" , df.info())
    print("\nMissing value count \n")
    print(df.isnull().sum())
    print("\nDuplicated value count" , df.duplicated().sum())
##önce datayıgörüntüle ve duplicate var mı flaan bak
#sonra 
# Örnek Kullanım
# preprocessor = DataPreprocessing(remove_punc=True, lowercase=True, remove_stops=True)
# sample_text = "Hello, World! This is another sample text for testing."
# processed_text = preprocessor.preprocess(sample_text)
# df['processed_text'] = df['text'].apply(processor.preprocess)

# print(processed_text) 
df = readDataset("training.1600000.processed.noemoticon.csv")
df_filtered = data_details(df)
preprocessor = DataPreprocessing(remove_punc=True, lowercase=True, remove_stops=True)
df['processed_text'] = df['text'].apply(preprocessor.preprocess)

print(df['processed_text'].head())








# def get_wordnet_pos(word):
#     tag = pos_tag([word])[0][1][0].upper()
#     #0 alırsan output running,vbg sonra 1 al index vbg sonra 0 al index v çüş:D
#     tag_dict = {
#         'J': wordnet.ADJ,
#         'N': wordnet.NOUN,
#         'V': wordnet.VERB,
#         'R': wordnet.ADV
#     }
#     return tag_dict.get(tag, wordnet.NOUN)

# def data_preprocessing(df):
#     df_filtered = df[['target' , 'text']]
#     print(df_filtered)

#     df_filtered.loc[:, 'text'] = df_filtered['text'].apply(lambda x: x.lower())
#     #loc meselesindeki esneklik şundan kaynaklı sadece belirli satırlara bile işlem ypaabilmene olanak sağlıyor :, kısmı tüm satırlar demek

#     df_filtered.loc[:, 'text'] = df_filtered['text'].replace('@[A-Za-z0-9]', '', regex=True)
#     df_filtered.loc[:, 'text'] = df_filtered['text'].replace(r'http\S+', '', regex=True)
#     df_filtered.loc[:, 'text'] = df_filtered['text'].replace(r'www\S+', '', regex=True)

#     #olum bunu yapmaya ne gerek var direkt üstünde değiştir işte sğdposfğpso 
#     texts = df_filtered.loc[:, 'text'].copy()
#     new_text = [re.sub('[0-9]+', '', text) for text in texts]
#     df_filtered.loc[:,'text'] = new_text

#     #bunu da daha optimize yazabiliriz x,y,z kısmı değiştirilecek,yerinegelen,silinecekse yukardakileri koyabiliriz mesela silinecek yere ve '' atar yerine?
#     #hayır maketrans sadece tekli karakterlerde çalışıyor..!
#     table = str.maketrans('', '', string.punctuation)
#     texts = df_filtered.loc[:,'text'].copy()
#     new_text = [' '.join([w.translate(table) for w in text.split()]) for text in texts]
#     df_filtered.loc[:,'text'] = new_text
#     df_filtered['text'].head(5)

#     # bu kısım words diye sütun oluşturup hepsini dizi gibi virgüllü yazıyo cümlelerin
#     df_filtered.loc[:, 'words'] = df_filtered['text'].apply(word_tokenize)
#     # sürekli aynı sütunları ekliyo gibi olduk azalta azalta hımhım 
#     #text --- words---- wordscleaned bunlar hep azaltıla azaltıla hepsi tutuluyo içinde

#     #ing kelimelerin olduğu byüüüükcene bi sözlük gibi düsün icinde olmayanları silcez
#     stop_words = set(stopwords.words('english'))

#     #burada tekrar lower yapmamız çok saçma yukarıda yapıyoruz zaten tekrar olmuş
#     df_filtered.loc[:, 'words_cleaned'] = df_filtered['words'].apply(lambda words: [word for word in words if word.lower() not in stop_words])
#     print(df_filtered[['text' , 'words', 'words_cleaned']].head())

    
#     #bu arada tak diye yeni sütun eklemesi pandastan geliyormus wo
#      # Lemmatization
#      #eğer türünü vermezsen kelimenin varsayılan olarak noun cevirir ki bu da runningi tekrar running olarak çevirir
#     lemmatizer = WordNetLemmatizer()
#     df_filtered['words_lemmatized'] = df_filtered['words_cleaned'].apply(
#         lambda words: [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
#     )

#     print(df_filtered[['words_cleaned' , 'words_lemmatized']].head(10))
    



# # df = readDataset("training.1600000.processed.noemoticon.csv")
# # df_filtered = data_details(df)
# # data_preprocessing(df)




# #bi kere indirsen yeterli projeyi başlatırken sonra yoruma al ama ganiz try excpti daha cok begenebilir.
# #NLTK (Natural Language Toolkit) 
# nltk.download('punkt_tab')
# try:
#     nltk.data.find('corpora/stopwords')
#     nltk.data.find('tokenizers/punkt')
#     nltk.data.find('corpora/wordnet')
#     nltk.data.find('taggers/averaged_perceptron_tagger')
# except LookupError:
#     nltk.download('stopwords')
#     nltk.download('punkt')
#     nltk.download('wordnet')
#     nltk.download('averaged_perceptron_tagger')