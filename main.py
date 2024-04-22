import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

path = r'data/amazon_cells_labelled.txt'
sentimental_df = pd.read_csv(path, names=('review', 'sentimental'), sep='\t')
reviews = sentimental_df['review'].values
sentiments = sentimental_df['sentimental'].values

reviews_train, reviews_test, sentiments_train, sentiments_test = train_test_split(reviews, sentiments, test_size=0.2,
                                                                                  random_state=500)
vectorize = CountVectorizer()
vectorize.fit(reviews)

X_train = vectorize.transform(reviews_train)
X_test = vectorize.transform(reviews_test)

classifier = LogisticRegression()
classifier.fit(X_train, sentiments_train)

accuracy = classifier.score(X_test, sentiments_test)

thought = input('Write your thought here: ')
new_reviews = list()
new_reviews.append(thought)

X_new = vectorize.transform(new_reviews)
predicts = classifier.predict(X_new)
results = [str(element) for element in ([(review_n, score_s)  for review_n, score_s in zip(new_reviews, predicts)])]
results = '\n'.join(results)
print(f"\tResults: \n{results}")