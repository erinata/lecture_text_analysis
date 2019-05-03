import json
import pandas as pd

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

dataset = dataset[(dataset['stars']==1)|(dataset['stars']==5)]

print(dataset.shape)

data = dataset['text']
target = dataset['stars']



