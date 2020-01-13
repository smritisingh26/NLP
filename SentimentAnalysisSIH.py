#!/usr/bin/env python
# coding: utf-8

# ### LOADING DATA 

# In[1]:


#Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import nltk.classify.util
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.classify import NaiveBayesClassifier
import numpy as np
import re
import string
import nltk
get_ipython().run_line_magic('matplotlib', 'inline')


# In[22]:


#Importing the Data
df=pd.read_csv('OneDrive\Desktop\Projects\WinNLP\AmazonRev.csv')
df.head()


# In[23]:


#Checking for null values in relevant columns
perm=df[['reviews.rating' , 'reviews.text' , 'reviews.title' , 'reviews.username']]
print(perm.isnull().sum())
perm.head()


# In[24]:


#Obtaining null values in relevant columns
check=df[df["reviews.rating"].isnull()]
check.head()


# In[61]:


#obtaining non null values in relevant columns
senti=perm[perm["reviews.rating"].notnull()]
perm.head()


# In[62]:


#Classification of text as positive and negative
senti["senti"]=senti["reviews.rating"]>=4
senti["senti"]=senti["senti"].replace([True , False] , ["pos" , "neg"])


# ### PREPROCESSING TEXT TO TRAIN MODEL

# In[64]:


import nltk.classify.util
from nltk.classify import NaiveBayesClassifier

cleanup_re=re.compile('[^a-z]+')
def cleanup(sentence):
    sentence=str(sentence)
    sentence=sentence.lower()
    sentence=cleanup_re.sub(' ', sentence).strip()
    sentence=" ".join(nltk.word_tokenize(sentence))
    return sentence

senti["Summary_Clean"]=senti["reviews.text"].apply(cleanup)
check["Summary_Clean"]=check["reviews.text"].apply(cleanup)


# In[66]:


#Splitting data into train and test set
split=senti[["Summary_Clean","senti"]]
train=split.sample(frac=0.8,random_state=200)
test=split.drop(train.index)


# In[68]:


#Feature Extractor 
def word_feats(words):
    features={}
    for word in words:
        features[word]=True
    return features

train["words"]=train["Summary_Clean"].str.lower().str.split()
test["words"]=test["Summary_Clean"].str.lower().str.split()
check["words"]=check["Summary_Clean"].str.lower().str.split()
train.index=range(train.shape[0])
test.index=range(test.shape[0])
check.index=range(check.shape[0])
prediction={}
train_naive = []
test_naive = []
check_naive = []

for i in range(train.shape[0]):
    train_naive = train_naive +[[word_feats(train["words"][i]) , train["senti"][i]]]
for i in range(test.shape[0]):
    test_naive = test_naive +[[word_feats(test["words"][i]) , test["senti"][i]]]
for i in range(check.shape[0]):
    check_naive = check_naive +[word_feats(check["words"][i])]
    
classifier = NaiveBayesClassifier.train(train_naive)
print("NLTK Naive bayes Accuracy : {}".format(nltk.classify.util.accuracy(classifier , test_naive)))
classifier.show_most_informative_features(5)


# In[69]:


#Predicting result of NLTK Classifier
y=[]
only_words=[test_naive[i][0] for i in range(test.shape[0])]
for i in range(test.shape[0]):
    y=y+[classifier.classify(only_words[i] )]
prediction["Naive"]= np.asarray(y)

y1=[]
for i in range(check.shape[0]):
    y1=y1+[classifier.classify(check_naive[i] )]
check["Naive"] = y1


# In[70]:


#Building TF-IDF Vectors and Count Vectors for data

from wordcloud import STOPWORDS
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
stopwords=set(STOPWORDS)
stopwords.remove("not")

count_vect=CountVectorizer(min_df=2 ,stop_words=stopwords , ngram_range=(1,2))
tfidf_transformer=TfidfTransformer()

X_train_counts=count_vect.fit_transform(train["Summary_Clean"])        
X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts)

X_new_counts=count_vect.transform(test["Summary_Clean"])
X_test_tfidf=tfidf_transformer.transform(X_new_counts)

checkcounts=count_vect.transform(check["Summary_Clean"])
checktfidf=tfidf_transformer.transform(checkcounts)


# In[71]:


#Using Logistic Regression to classify 
from sklearn import linear_model
logreg=linear_model.LogisticRegression(solver='lbfgs' , C=1000)
logistic=logreg.fit(X_train_tfidf, train["senti"])
prediction['LogisticRegression']=logreg.predict_proba(X_test_tfidf)[:,1]
print("Logistic Regression Accuracy : {}".format(logreg.score(X_test_tfidf , test["senti"])))

check["log"]=logreg.predict(checktfidf)


# In[72]:


#Obtaining Most Commonly Occuring Words
words=count_vect.get_feature_names()
feature_coefs=pd.DataFrame(data = list(zip(words, logistic.coef_[0])),columns = ['feature', 'coef'])
feature_coefs.sort_values(by="coef")


# In[79]:


def test_sample(model, sample):
    sample_counts = count_vect.transform([sample])
    sample_tfidf = tfidf_transformer.transform(sample_counts)
    result = model.predict(sample_tfidf)[0]
    prob = model.predict_proba(sample_tfidf)[0]
    print("Sample estimated as %s: negative prob %f, positive prob %f" % (result.upper(), prob[0], prob[1]))

test_sample(logreg, "The product was good and easy to  use")
test_sample(logreg, "the whole experience was horrible and product is worst")
test_sample(logreg, "product is not good")
test_sample(logreg, "Bad product value.")
test_sample(logreg, "I'm really impressed with the quality!")


# In[ ]:




