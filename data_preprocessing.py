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
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from scipy import stats


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')


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


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

def plot_workclouds(positive_text, negative_text):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Generate the wordcloud for positive tweets
    wordcloud_positive = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(positive_text)
    axes[0].imshow(wordcloud_positive, interpolation='bilinear')
    axes[0].set_title('Positive tweets')
    axes[0].axis('off')

    # Generate the wordcloud for negative tweets
    wordcloud_negative = WordCloud(width=800, height=400, background_color='white', colormap='magma').generate(negative_text)
    axes[1].imshow(wordcloud_negative, interpolation='bilinear')  # Fixed here: use wordcloud_negative
    axes[1].set_title('Negative tweets')
    axes[1].axis('off')

    plt.subplots_adjust(wspace=0.2)
    plt.show()

def data_preprocessing(df):
    df_filtered = df[['target' , 'text']]
    print(df_filtered)

    df_filtered = df[['target', 'text']].copy()
    df_filtered['text'] = df_filtered['text'].str.lower()
    df_filtered['text'] = df_filtered['text'].replace('@[A-Za-z0-9]+', '', regex=True)
    df_filtered['text'] = df_filtered['text'].replace(r'htt\S+', '', regex=True)
    df_filtered['text'] = df_filtered['text'].replace(r'www\S+', '', regex=True)
    df_filtered['text'] = df_filtered['text'].apply(lambda x: re.sub('[0-9]+', '', x))
    table = str.maketrans('', '', string.punctuation)
    df_filtered['text'] = df_filtered['text'].apply(lambda x: ' '.join([w.translate(table) for w in x.split()]))
    df_filtered['words'] = df_filtered['text'].apply(word_tokenize)
    stop_words = set(stopwords.words('english'))
    df_filtered['words_cleaned'] = df_filtered['words'].apply(lambda words: [word for word in words if word.lower() not in stop_words])
    print(df_filtered[['text', 'words', 'words_cleaned']].head())

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    df_filtered['words_lemmatized'] = df_filtered['words_cleaned'].apply(
         lambda words: [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
    )
    print()
    print(df_filtered[['words_cleaned' , 'words_lemmatized']].head(10))

    # Remove extra spaces
    df_filtered['processed_text'] = df_filtered['words_lemmatized'].apply(lambda words: ' '.join(words).strip())
    print(df_filtered['processed_text'].head())

    # Remove rare word
    all_words = ' '.join(df_filtered['processed_text']).split()
    word_freq = pd.Series(all_words).value_counts()
    less_frequent = word_freq[word_freq == 1].index
    df_filtered['processed_text'] = df_filtered['processed_text'].apply(lambda x: ' '.join(word for word in x.split() if word not in less_frequent))

    print(df_filtered[['text', 'processed_text']].head())

    # duplicated rows
    duplicated_rows = df_filtered[df_filtered['processed_text'].duplicated()]
    print(f"Duplicated rows: {duplicated_rows.shape[0]}")

    df_filtered_unique = df_filtered.drop_duplicates(subset='processed_text')

    print(f"Original dataset size: {df_filtered.shape[0]}")
    print(f"After remove duplicated rows dataset size: {df_filtered_unique.shape[0]}")

    #update DataFrame
    df_filtered = df_filtered_unique

    # target variable adjustment:

    #change target values positive 4 to 1
    df_filtered['target'] = df_filtered['target'].replace(4,1)
    print(df_filtered['target'].head())

    #Converting texts to Numerical Values:

    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(df_filtered['processed_text'])
    print(df_filtered['processed_text'].apply(type).value_counts())


    # DATA VISUALIZATION:

    #target distribution:
    target_counts = df_filtered['target'].value_counts()

    labels = {
        0: 'Negative',
        1: 'Positive'
    }

    custom_labels = [labels.get(x, str(x)) for x in target_counts]

    colors = ['#2b8cbe', '#7bccc4']

    plt.figure(figsize=(7,7))
    plt.pie(target_counts, labels=custom_labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Target Distribution', size=14)
    plt.show()

    positive_text = ' '.join(df_filtered[df_filtered['target'] == 1]['processed_text'].astype(str))
    negative_text = ' '.join(df_filtered[df_filtered['target'] == 0]['processed_text'].astype(str))

    plot_workclouds(positive_text, negative_text)

    # group lengths by emotion labels
    df_filtered['text_length'] = df_filtered['processed_text'].apply(len)

    plt.figure(figsize=(12,6))
    for sentiment in df_filtered['target'].unique():
        sentiment_label = 'Positive' if sentiment == 0 else 'Negative'
        subset = df_filtered[df_filtered['target'] == sentiment]
        color = '#2b8cbe' if sentiment == 0 else '#238b45'
        plt.hist(subset['text_length'], bins=50, alpha=0.5, label=f'Sentiment {sentiment_label}' , color=color)

    plt.title('Relationship Between Sentiment and Tweet Length')
    plt.xlabel('Tweet Length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    #sentiment distribution per user

    top_users = df['user'].value_counts().head().index
    top_users_df = df[df['user'].isin(top_users)]

    user_sentiment = top_users_df.groupby('user')['target'].value_counts().unstack()

    #visualization
    user_sentiment.plot(kind='bar', stacked=True, figsize=(8,5), color=['#2b8cbe', '#cb181d'])
    plt.title('Sentiment Distrubition of Most Popular Users')
    plt.xlabel('Users')
    plt.ylabel('Tweets')
    plt.legend(title='Sentiment', labels=['Positive' , 'Negative'])
    plt.xticks(rotation=45)
    plt.show()

    # Outlier
    z_scores = stats.zscore(df_filtered['text_length'])
    outliers = df_filtered[abs(z_scores) > 3]
    print(outliers)

    # Distribution Graph
    plt.figure(figsize=(8,5))
    plt.scatter(df_filtered.index, df_filtered['text_length'], color='blue', label='Tweet Lengths')
    plt.scatter(outliers.index, outliers['text_length'], color='red', label='Outliers', marker='o', s=100)
    plt.title('Tweet Lengths and Outliers')
    plt.xlabel('Index')
    plt.ylabel('Tweet Lengths')
    plt.legend()
    plt.grid(True)
    plt.show()
  

df = readDataset("training.1600000.processed.noemoticon.csv")
#df = readDataset("test_dataset.csv")
data_details(df)
data_preprocessing(df)




