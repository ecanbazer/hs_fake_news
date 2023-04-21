#https://www.kaggle.com/datasets/nopdev/real-and-fake-news-dataset

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('../fake_or_real_news.csv')
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size = 0.3, random_state = 41)

count_vectorizer = CountVectorizer(stop_words = 'english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer(stop_words = 'english')
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Import the necessary modules
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(count_train,y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
score_count = metrics.accuracy_score(y_test, pred)


# Create a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier_tfidf = MultinomialNB()

# Fit the classifier to the training data
nb_classifier_tfidf.fit(tfidf_train, y_train)

# Create the predicted tags: pred
pred_tfidf = nb_classifier_tfidf.predict(tfidf_test)

# Calculate the accuracy score: score
score_tfidf = metrics.accuracy_score(y_test, pred_tfidf)

print("Model: MultinomialNB()")
print("Vectorizer: CountVectorizer")
print("Accuracy: " + str(score_count))
print()
print("Model: MultinomialNB()")
print("Vectorizer: TfidfVectorizer")
print("Accuracy: " + str(score_tfidf))