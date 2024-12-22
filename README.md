# Sentiment Analysis and Preprocessing - README

## Dataset
```text
Source: [Kaggle - Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140/data)
```

## Project Overview
```text
This project focuses on sentiment analysis of textual data (e.g., tweets). It includes:
- Data preprocessing
- Visualization
- Feature extraction
- Classification using multiple machine learning models
The pipeline ensures efficient data preparation and analysis, making it suitable for sentiment detection tasks.
```

## Requirements
```text
To run this project, install the following libraries:
- Data Handling: pandas, numpy
- Visualization: matplotlib, seaborn, wordcloud
- Natural Language Processing (NLP): nltk, textblob
- Machine Learning: scikit-learn
- Other Dependencies: scipy

Install these libraries using:
```bash
pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn scipy textblob
```
```

## Steps
```text
1. Data Reading
   - The `readDataset` function reads a CSV file and extracts the middle 10,000 rows for analysis.
   - Column names: ['target', 'ids', 'date', 'flag', 'user', 'text'].

2. Exploratory Data Analysis (EDA)
   - Use `data_details` to identify summary statistics, missing values, and duplicates.
   - Detect outliers in textual lengths using Z-scores.

3. Preprocessing
   - **Cleaning Text:** Convert text to lowercase. Remove URLs, mentions, digits, and punctuation. Tokenize text and remove stopwords.
   - **Lemmatization:** Standardize words to their base form using WordNetLemmatizer.
   - **Rare Word Removal:** Exclude words that occur only once in the dataset.
   - **Vectorization:** Transform text into numerical values using TfidfVectorizer.

4. Visualization
   - Generate word clouds for positive and negative sentiments.
   - Visualize text length distribution by sentiment.
   - Analyze sentiment distribution for popular users.
   - Detect and visualize outliers.

5. Classification Models
   - Evaluate the following models for sentiment classification:
     - Logistic Regression
     - Random Forest
     - Support Vector Classifier (SVC)
     - Naive Bayes
     - Decision Tree (Gini/Entropy)
     - Artificial Neural Networks (ANN)
     - K-Nearest Neighbors (KNN)

6. Model Evaluation
   - Perform train-test split with an 80-20 ratio.
   - Evaluate models using metrics such as accuracy, precision, recall, F1-score, and confusion matrices.
```

## Usage
```text
1. Load the dataset using the `readDataset` function.
2. Explore the dataset and handle missing or duplicated values using `data_details`.
3. Preprocess the text data using `data_preprocessing`.
4. Visualize insights using the provided plotting functions.
5. Train machine learning models and evaluate their performance.
```

## Notes
```text
- NLP Resources: The project downloads required NLTK resources like stopwords and POS taggers.
