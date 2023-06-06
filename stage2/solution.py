from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics
import pandas as pd

df = pd.read_csv('../fake_or_real_news.csv')
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size = 0.3, random_state = 41)

count_vectorizer = CountVectorizer(stop_words = 'english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
svc = SVC(random_state = 41)

# Fit the classifier to the training data
svc.fit(count_train,y_train)

# Create the predicted tags: pred
y_pred_svc = svc.predict(count_test)

# Calculate the accuracy score: score
score_svc = metrics.accuracy_score(y_test, y_pred_svc)


# Create a Multinomial Naive Bayes classifier: nb_classifier
logreg = LogisticRegression(random_state = 41)

# Fit the classifier to the training data
logreg.fit(count_train, y_train)

# Create the predicted tags: pred
y_pred_logreg = logreg.predict(count_test)

# Calculate the accuracy score: score
score_logreg = metrics.accuracy_score(y_test, y_pred_logreg)

print("Model: SVC()")
print("Accuracy: " + str(score_svc))
print()
print("Model: LogisticRegression()")
print("Accuracy: " + str(score_logreg))