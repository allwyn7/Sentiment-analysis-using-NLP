import numpy as np
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t', quoting = 3)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
  review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
  review = review.lower()
  review = review.split()
  all_stopwords = set(stopwords.words('english'))
  all_stopwords.remove('not')
  ps = PorterStemmer()
  review = [ps.stem(j) for j in review if not j in all_stopwords]
  review = ' '.join(review)
  corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)

new_review = 'I hate this restaurant-'
new_review = re.sub('[^a-zA-Z]',' ',new_review)
new_review = new_review.lower()
new_review = new_review.split()
new_review = [ps.stem(i) for i in new_review if not i in all_stopwords]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_x_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_x_test)

print(new_y_pred)