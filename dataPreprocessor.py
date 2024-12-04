import re
import string
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet



#burada bu sınıfta yapacaklarımı topluyprum fakat init kısmında daha farklı datalara 
#uygulanabilmesi için böyle bir sistem kurduk farklı datalarda özel karakterleri kaldırmak istemeyeiblirsin
#daha uygulanabilirliği devamlılığı artan bi kod??hem de ön işleme fonksiyonları ful burda
class DataPreprocessing:
    def __init__ (self, remove_punc=False, lowercase=False,
                  remove_stops=False, lemmatize=True,
                  remove_mentions=False, remove_urls=False,remove_numbers=False):
        self.remove_mentions = remove_mentions
        self.remove_urls = remove_urls
        self.remove_numbers = remove_numbers
        self.remove_punc= remove_punc
        self.lowercase= lowercase
        self.remove_stops= remove_stops
        self.lemmatize = lemmatize
        self.stop_words= set(stopwords.words('english'))
        self.table= str.maketrans('','',string.punctuation)
        


    #böyle yaparak datayı ful buraya taşıdık ağır olabilir diye düşündüm ama
    #sadece dataframe nesnesinin referansını taşıyormuşuz ve memorylik bir şey yokmuş
    #o yüzden ağır değilmiş
    #pipeline hizmeti hem de sadece istenilenleri yapması için if else
    def preprocess(self, text):
        if self.remove_mentions:
            text = self.remove_mentions_from_text(text)
        if self.lowercase:
            text = self.to_lowercase(text)
        if self.remove_urls:
            text = self.remove_urls_from_text(text)
        if self.remove_numbers:
            text = self.remove_numbers_from_text(text)
        if self.remove_punc:
            text = self.remove_punctuation(text)
        if self.remove_stops:
            text = self.remove_stopwords(text)
        print(text)
        if self.lemmatize:
            text = self.lemmatize_words(text)
        print(text)
        
        # Return a list of words
        return text.split()  # This splits the text into words


    def remove_punctuation(self, text):
        """Remove punctuation from text."""
        table = str.maketrans('', '', string.punctuation)
        return ' '.join([word.translate(table) for word in text.split()])

    def to_lowercase(self, text):
        return text.lower()

    def remove_stopwords(self, text):
        """Remove stopwords from text."""
        stop_words = set(stopwords.words('english'))
        return ' '.join([word for word in text.split() if word.lower() not in stop_words])
    
    def lemmatize_words(self, text):
        """Lemmatize words in text."""
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(text)
        return ' '.join([lemmatizer.lemmatize(word, self.get_wordnet_pos(word)) for word in words])
    
    def get_wordnet_pos(self, word):
        """Get the part of speech tag for a word."""
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    def remove_mentions_from_text(self, text):
        """@ ile başlayan kullanıcı adlarını kaldırır."""
        return re.sub(r'@[A-Za-z0-9_]+', '', text)

    def remove_urls_from_text(self, text):
        """URL'leri kaldırır."""
        text = re.sub(r'http\S+', '', text)  # http URL'lerini kaldırır
        text = re.sub(r'www\S+', '', text)   # www URL'lerini kaldırır
        return text

    def remove_numbers_from_text(self, text):
        """Sayılardan metni temizler."""
        return re.sub(r'[0-9]+', '', text)


 