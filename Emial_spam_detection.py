#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the dataset
df = pd.read_csv("D:\RAHUL - Copy\DATA SETS\spam.csv", encoding='latin-1')

# Display the first few rows of the dataframe
df.head()


# In[2]:


# Check for missing values and get a summary of the dataset
df.info()
print('\
Missing values in each column:')
print(df.isnull().sum())


# In[4]:


from sklearn.model_selection import train_test_split

# Encode the 'Category' column
df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})

# Split the dataset into features and labels
X = df['Message']
y = df['Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Data preprocessing completed. Dataset split into training and testing sets.')


# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_tfidf = vectorizer.transform(X_test)

print('Feature engineering completed. Text data transformed into TF-IDF features.')


# In[7]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Initialize the Naive Bayes classifier
nb_classifier = MultinomialNB()

# Train the classifier on the TF-IDF features
nb_classifier.fit(X_train_tfidf, y_train)

# Predict on the test data
y_pred = nb_classifier.predict(X_test_tfidf)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print(f'Model training completed. Accuracy: {accuracy:.2f}')


# In[8]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}

# Initialize the Grid Search model
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')

# Fit the model
grid_search.fit(X_train_tfidf, y_train)

# Best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f'Best Parameters: {best_params}')
print(f'Best Score: {best_score:.2f}')


# In[9]:


from joblib import dump

# Save the optimized model
model_filename = 'optimized_spam_classifier.joblib'
dump(grid_search.best_estimator_, model_filename)

print(f'Model saved as {model_filename}')


# In[ ]:




