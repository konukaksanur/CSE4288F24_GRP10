from dataPreprocessor import DataPreprocessing

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()



preprocessor = DataPreprocessing(lemmatize=True)
text = "The quick brown foxes are running"
processed_text = preprocessor.preprocess(text)
print(processed_text)
