import pandas as pd

# separating fruits
fruits = pd.read_csv('heart.csv', sep='\t')
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))

fruits.head()

X = fruits[['height', 'width', 'mass', 'color_score']]
y = fruits['fruit_label']

from sklearn.model_selection import train_test_split
#random_state: set seed for random# generator
#test_size: default 25% testing, 75% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

# Create a linear model : LogisticRegression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=40, max_iter=10000)
lr.fit(X_train, y_train)

# Coefficients of linear model (b_1,b_2,...,b_p): log(p/(1-p)) = b0+b_1x_1+b_2x_2+...+b_px_p
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))

# Estimate the accuracy of the classifier on future data, using the test data
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

# Use the trained logistic regression classifier model to classify new, previously unseen objects
# first example: a small fruit with mass 15g, color_score = 5.5, width 4.3 cm, height 5.5 cm
testFruit = pd.DataFrame([[5.5, 4.3, 15, 5.5]], columns=['height', 'width', 'mass', 'color_score'])
fruit_prediction = lr.predict(testFruit)
print(lookup_fruit_name[fruit_prediction[0]])

predict_probability = lr.predict_proba(testFruit)
print("predict: {}".format(predict_probability[0]))

# second example: a larger one
testFruit = pd.DataFrame([[8.5, 6.3, 190, 0.53]], columns=['height', 'width', 'mass', 'color_score'])
fruit_prediction = lr.predict(testFruit)
print(lookup_fruit_name[fruit_prediction[0]])

predict_probability = lr.predict_proba(testFruit)
print("predict: {}".format(predict_probability[0]))
