#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, pandas as pd, nltk

# from sklearn import datasets
# from datetime import datetime

path = "C:/Users/Shoaib/Desktop/Dataset.csv"

df = pd.read_csv(path, encoding = "latin1")

df["Comments"] = df["Comments"].str.lower()
df["Comments"]


# In[2]:


shape = df.shape
shape
Original_len = len(df)
Original_len


# ## Pre- Processing

# In[3]:


#removing HTML codes and <href>

import re
def remove_html(text):
    pattern = re.compile("<,*?>") 
    return pattern.sub(" ", text)

df["Comments"] = df["Comments"].apply(remove_html)
df["Comments"]


# In[4]:


#removing URl from the comments

def remove_urls(text):
    pat = re.compile(r'https?://\S+|www\.\S+')
    cleaned_text = re.sub(pat, '', text)

    return cleaned_text

df["Comments"] = df["Comments"].apply(remove_urls)

middle_rows = df["Comments"].iloc[len(df)//2 - 4:len(df)//2 + 4]
print(middle_rows)


# In[5]:


# removing punctuation from the comments

import string
remove = string.punctuation 

def remove_punc(text):
    return text.translate(str.maketrans(" "," ", remove))

df["Comments"] = df["Comments"].apply(remove_punc)
df["Comments"]


# In[6]:


#spelling checking and correction

get_ipython().system('pip install textblob')
from textblob import TextBlob

for comment in df["Comments"]:
    tb = TextBlob("Comment")
    tb.correct().string
    
df["Comments"]


# In[7]:


# removing integers

def remove_integers(text):
    int_pattern = r'[0-9]+'
    cleaned_text = re.sub(int_pattern, '', text)
    return cleaned_text

df["Comments"]= df["Comments"].apply(remove_integers)
df["Comments"]


# In[8]:


#removing stop words and preposition

from nltk.corpus import stopwords
nltk.download("stopwords")

stopwords.words("english")

def remove_stopwords(text):
    new_text = []
    for word in text.split():
        if word in stopwords.words("english"):
            new_text.append(" ")
        else:
            new_text.append(word)
    
    cleaned_text = new_text[:]
    new_text.clear()
    return " ".join(cleaned_text)

df["Comments"] = df["Comments"].apply(remove_stopwords)
df["Comments"]


# In[9]:


#removing duplicate values from the dataset
df["Comments"] = df["Comments"].str.replace('hahah', '')
df["Comments"] = df["Comments"].drop_duplicates()

# removing NaN values
df.dropna(inplace = True)
df["Comments"]


# In[10]:


#number of rows/comments drop in missing values and duplicates

New_dataset_len = len(df)
New_dataset_len

# Display the number of values dropped
values_dropped = Original_len - New_dataset_len
values_dropped
print(f"Number of values dropped: {values_dropped}")


# In[11]:


#removing other language words from comments data

# def remove_non_english(text):    
#      english_pattern = re.compile(r'[^\x00-\x7F]')
#      cleaned_text = english_pattern.sub('', df['Text'].str.replace("", ))
#      return cleaned_text

english_pattern = re.compile(r'[^\x00-\x7F]')

# Perform regular expression operation on the 'Text' column using str.replace()
df["Comments"] = df["Comments"].str.replace(r'english_pattern', '')
df["Comments"]


# In[12]:


# removing emoji
def remove_emojis(text):
    if isinstance(text, str):
        return re.sub(r'[^\x00-\x7F]+', '', text)
    else:
        return text

df["Comments"] = df["Comments"].apply(remove_emojis)
df["Comments"]


# In[13]:


#Tokenization

# def tokenize_text(text):
#     if isinstance(text, str):
#         doc = nlp(text)
#         return [token.text for token in doc]
#     else:
#         return []

# df['Tokens'] = df["Comments"].apply(tokenize_text)
# df['Tokens']

# df["Comments"]

# # tokenziation of NLTK take string text
# from nltk.tokenize import word_tokenize, sent_tokenize
# nltk.download("punkt")

# df['comments'] = df['comments'].astype(str)
# df["comments"] = df["comments"].apply(lambda x: word_tokenize(x))
# df["comments"]



get_ipython().system('pip install spacy')
import spacy
get_ipython().system('python -m spacy download en_core_web_sm')
nlp = spacy.load("en_core_web_sm")


# Function to tokenize a comment using spaCy
def tokenize_comment(Comments):
    doc = nlp(Comments)
    return [token.text for token in doc]

# Apply tokenization to the 'comments' column and create a new 'tokens' column
df['tokens'] = df['Comments'].apply(tokenize_comment)

df['tokens']
df['Comments']


# In[14]:


# #lemmatization

from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
nltk.download('omw')
nltk.download('omw-1.4')

wordnet_lemmatizer = WordNetLemmatizer()

print("{0:20}{1:20}".format("word", "Lemma"))

# for word in df['tokens']:
#      print("{0:20}{1:20}".format(word, wordnet_lemmatizer.lemmatize(word, pos= 'v')))


# Function to perform lemmatization on a comment using spaCy
def lemmatize_comment(Comment):
    doc = nlp(Comment)
    return ' '.join([token.lemma_ for token in doc])

# Apply lemmatization to the 'comments' column and create a new 'lemmatized_comments' column
df['lemmatized_comments'] = df['Comments'].apply(lemmatize_comment)

# Display the resulting DataFrame
print(df)


# # feature extraction

# In[24]:


# Feature extraction via TF-IDF uni-gram

from sklearn.feature_extraction.text import TfidfVectorizer
Tfidf1 = TfidfVectorizer()

Tfidf1.fit_transform(df["Comments"]).toarray()
print(Tfidf1.idf_)


# In[25]:


# Feature extraction via TF-IDF bi-gram

Tfidf2 = TfidfVectorizer(ngram_range = (2,2))

Tfidf2.fit_transform(df["Comments"]).toarray()
print(Tfidf2.idf_)


# In[26]:


# Feature extraction via TF-IDF tri-gram

Tfidf3 = TfidfVectorizer(ngram_range = (3, 3))
Tfidf3.fit_transform(df["Comments"]).toarray()
print(Tfidf3.idf_)


# In[27]:


# bigram using count vectorizer bag of words

from sklearn.feature_extraction.text import CountVectorizer
CV = CountVectorizer(ngram_range = (2, 2))
BOW = CV.fit_transform(df["Comments"])

# print(CV.vocabulary_)
BOW[100:300].toarray()


# In[28]:


# Naive Bayes multinomialNB using uni-gram features

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Spliting the data into features comments and labels
X = df["Comments"]
y = df["Classes"]

# Convert text data into feature vectors using TF-IDF
X1 = Tfidf1.fit_transform(X)

# Split the data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.34, random_state=42)

# Create a Multinomial Naive Bayes model
naive_bayes_model = MultinomialNB()

# Train the model
naive_bayes_model.fit(X1_train, y1_train)

# Make predictions on the testing data
y_pred1 = naive_bayes_model.predict(X1_test)

# Evaluate the model
accuracy = accuracy_score(y1_test, y_pred1)
report = classification_report(y1_test, y_pred1)
confusion_mat = confusion_matrix(y1_test, y_pred1)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
print("Confusion Matrix: ")
print(confusion_mat)


# In[30]:


# Naive Bayes multinomialNB using bi-gram features


# Convert text data into feature vectors using TF-IDF
X2 = Tfidf2.fit_transform(X)

# Split the data into training and testing sets
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.34, random_state=42)

# Create a Multinomial Naive Bayes model
naive_bayes_model = MultinomialNB()

# Train the model
naive_bayes_model.fit(X2_train, y2_train)

# Make predictions on the testing data
y_pred2 = naive_bayes_model.predict(X2_test)

# Evaluate the model
accuracy2 = accuracy_score(y2_test, y_pred2)
report2 = classification_report(y2_test, y_pred2)
confusion_mat2 = confusion_matrix(y2_test, y_pred2)

print(f"Accuracy: {accuracy2}")
print("Classification Report:")
print(report2)
print("Confusion Matrix: ")
print(confusion_mat2)


# In[31]:


# Naive Bayes multinomialNB by tri-gram

# Convert text data into feature vectors using TF-IDF
X3 = Tfidf3.fit_transform(X)

# Split the data into training and testing sets
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y, test_size=0.34, random_state=42)

# Create a Multinomial Naive Bayes model
naive_bayes_model = MultinomialNB()

# Train the model
naive_bayes_model.fit(X3_train, y3_train)

# Make predictions on the testing data
y_pred3 = naive_bayes_model.predict(X3_test)

# Evaluate the model
accuracy3 = accuracy_score(y3_test, y_pred3)
report3 = classification_report(y3_test, y_pred3)
confusion_mat3 = confusion_matrix(y3_test, y_pred3)

print(f"Accuracy: {accuracy3}")
print("Classification Report:")
print(report3)
print("Confusion matrix: ")
print(confusion_mat3)


# In[34]:


# Radial Basis Function
from sklearn.svm import SVC


# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=42)

# Creating a TF-IDF vectorizer to train the model
Tfidf_RBF = TfidfVectorizer()
X_train_tfidf = Tfidf_RBF.fit_transform(X_train)
X_test_tfidf = Tfidf_RBF.transform(X_test)

# Initializing the SVM classifier with RBF kernel
svm_classifier = SVC(kernel='rbf', gamma='scale', C=1.0)

# Train the classifier on the training data
svm_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = svm_classifier.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate a classification report (optional)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[36]:


from sklearn.ensemble import RandomForestClassifier

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer to convert text data into numerical vectors
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate a classification report (optional)
print("Classification Report:")
print(classification_report(y_test, y_pred))


