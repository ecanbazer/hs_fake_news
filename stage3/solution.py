from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import VotingClassifier
from sklearn import metrics
import pandas as pd

df = pd.read_csv('/Users/emrecanbazer/Desktop/python-things/fake_news_classification/hs_fake_news/fake_or_real_news.csv')
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size = 0.3, random_state = 41)

df['text']

count_vectorizer = CountVectorizer(stop_words = 'english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)


nb_classifier = MultinomialNB()
svc = SVC(random_state = 41)
logreg = LogisticRegression(random_state = 41)

classifiers = [('Naive Bayes Classifier', nb_classifier), ('State Vector Classifier', svc), ('Logistic Regression', logreg)]

vc = VotingClassifier(estimators=classifiers)

vc.fit(count_train,y_train)
y_pred = vc.predict(count_test)
acc = metrics.accuracy_score(y_test, y_pred)

print("Accuracy: " + str(acc))