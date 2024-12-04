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
    def __init__ (self, remove_punc=True, lowercase=True,
                  remove_stops=True , lemmatize=True,
                  remove_mentions=True , remove_urls=True ,remove_numbers=True):
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
    def preprocess(self, df, column_name):
    # Iterate over the DataFrame rows and process each text
        for index, row in df.iterrows():
            text = row[column_name]

            # Create a dictionary to hold results for each step
            processed_text = text

            # Process each step and add the result to the DataFrame
            if self.remove_mentions:
                processed_text = self.remove_mentions_from_text(processed_text)
                df.at[index, f'{column_name}_mentions_removed'] = processed_text
            if self.lowercase:
                processed_text = self.to_lowercase(processed_text)
                df.at[index, f'{column_name}_lowercase'] = processed_text
            if self.remove_urls:
                processed_text = self.remove_urls_from_text(processed_text)
                df.at[index, f'{column_name}_urls_removed'] = processed_text
            if self.remove_numbers:
                processed_text = self.remove_numbers_from_text(processed_text)
                df.at[index, f'{column_name}_numbers_removed'] = processed_text
            if self.remove_punc:
                processed_text = self.remove_punctuation(processed_text)
                df.at[index, f'{column_name}_punctuation_removed'] = processed_text
            if self.remove_stops:
                processed_text = self.remove_stopwords(processed_text)
                df.at[index, f'{column_name}_stopwords_removed'] = processed_text
            if self.lemmatize:
                processed_text = self.lemmatize_words(processed_text)
                df.at[index, f'{column_name}_lemmatized'] = processed_text

        # Return the updated DataFrame with new columns
        return df



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


 