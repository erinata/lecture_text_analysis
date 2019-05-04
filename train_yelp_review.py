import json
import pandas as pd
import string

import nltk

from nltk.corpus import stopwords
# nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,accuracy_score

review_text = []
review_stars = []
with open('yelp_review_part.json') as f:
    for line in f:
        json_line = json.loads(line)
        review_text.append(json_line["text"])
        review_stars.append(json_line["stars"])

dataset = pd.DataFrame(data = {'text': review_text, 'stars' : review_stars})#, columns=['text', 'stars'])

print(dataset.shape)

dataset = dataset[0:3000]

print(dataset.shape)

dataset = dataset[(dataset['stars']==1)|(dataset['stars']==3)|(dataset['stars']==5)]

print(dataset.shape)

data = dataset['text']
target = dataset['stars']

lemmatizer = WordNetLemmatizer()

def pre_processing(text):
	text_processed = [char for char in text if char not in string.punctuation]
	text_processed = ''.join(text_processed)
	return [lemmatizer.lemmatize(word.lower()) for word in text_processed.split() if word.lower() not in stopwords.words('english')]


print(pre_processing("This is some text. Hello!!! This is pretending to be a review! Reviews are funny." ))

count_vectorize_transformer = CountVectorizer(analyzer=pre_processing).fit(data)

print(count_vectorize_transformer.get_feature_names())

data = count_vectorize_transformer.transform(data)

data_training, data_test, target_training, target_test = train_test_split(data, target, test_size = 0.25)

machine = MultinomialNB()

machine.fit(data_training, target_training)

predictions = machine.predict(data_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))

test_review  = "It's a horrible resturant. It's expensive!!!!"
test_review_transformed = count_vectorize_transformer.transform([test_review])
prediction = machine.predict(test_review_transformed)
prediction_prob = machine.predict_proba(test_review_transformed)
print(prediction)
print(prediction_prob)


test_review  = "Baby Shark Duh duh duh duh duh"
test_review_transformed = count_vectorize_transformer.transform([test_review])
prediction = machine.predict(test_review_transformed)
prediction_prob = machine.predict_proba(test_review_transformed)
print(prediction)
print(prediction_prob)




