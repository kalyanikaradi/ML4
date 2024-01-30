#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import string
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("spam.tsv",sep='\t',names=['Class','Message'])
data.head(8) 


# In[3]:


# to view the first record
data.loc[:0]


# In[4]:


# Summary of the dataset
data.info()


# In[5]:


# create a column to keep the count of the characters present in each record
data['Length'] = data['Message'].apply(len)


# In[6]:


data['Length']


# In[7]:


data.head()


# In[8]:


## The mails are categorised into 2 classes ie., spam and ham. 
# Let's see the count of each class
data.groupby('Class').count()


# In[9]:


#Data Visualization
data['Length'].describe() # to find the max length of the message. 


# In[10]:


data['Length']==910


# In[11]:


# the message that has the max characters
data[data['Length']==910]['Message']


# In[12]:


# view the message that has 910 characters in it
data[data['Length']==910]['Message'].iloc[0]


# In[13]:


# View the message that has min characters
data[data['Length']==2]['Message'].iloc[0]


# In[14]:


# creating an object for the target values
dObject = data['Class'].values
dObject


# In[15]:


# Lets assign ham as 1
data.loc[data['Class']=="ham","Class"] = 1


# In[16]:


# Lets assign spam as 0
data.loc[data['Class']=="spam","Class"] = 0


# In[17]:


dObject2=data['Class'].values
dObject2


# In[18]:


data.head(8)


# In[19]:


# the default list of punctuations
import string

string.punctuation


# In[20]:


# Why is it important to remove punctuation?

"This message is spam" == "This message is spam."


# In[21]:


# Let's remove the punctuation

def remove_punct(text):
    text = "".join([char for char in text if char not in string.punctuation])
    return text

data['text_clean'] = data['Message'].apply(lambda x: remove_punct(x))

data.head()


# In[22]:


data.head(8)


# In[23]:


# Countvectorizer is a method to convert text to numerical data. 

# Initialize the object for countvectorizer 
CV = CountVectorizer(stop_words="english")


# In[24]:


# Splitting x and y

xSet = data['text_clean'].values
ySet = data['Class'].values
ySet


# In[25]:


# Datatype for y is object. lets convert it into int
ySet = ySet.astype('int')
ySet


# In[26]:


xSet


# In[27]:


xSet_train,xSet_test,ySet_train,ySet_test = train_test_split(xSet,ySet,test_size=0.2, random_state=10)


# In[28]:


xSet_train_CV = CV.fit_transform(xSet_train)
xSet_train_CV


# In[29]:


# Training a model
# With messages represented as vectors, we can finally train our spam/ham classifier. Now we can actually use almost any sort of classification algorithms. For a variety of reasons, the Naive Bayes classifier algorithm is a good choice.

# Initialising the model
NB = MultinomialNB()


# In[30]:


# feed data to the model
NB.fit(xSet_train_CV,ySet_train)


# In[31]:


# Let's test CV on our test data
xSet_test_CV = CV.transform(xSet_test)


# In[32]:


# prediction for xSet_test_CV

ySet_predict = NB.predict(xSet_test_CV)
ySet_predict


# In[33]:


# Checking accuracy

accuracyScore = accuracy_score(ySet_test,ySet_predict)*100

print("Prediction Accuracy :",accuracyScore)


# In[34]:


msg = input("Enter Message: ") # to get the input message
msgInput = CV.transform([msg]) # 
predict = NB.predict(msgInput)
if(predict[0]==0):
    print("------------------------MESSAGE-SENT-[CHECK-SPAM-FOLDER]---------------------------")
else:
    print("---------------------------MESSAGE-SENT-[CHECK-INBOX]------------------------------")


# In[36]:


# creating a list of sentences
documents = ["Dog bites man.", "Man bites dog.", "Dog eats meat.", "Man eats food."]

# Changing the text to lower case and remove the full stop from text
processed_docs = [doc.lower().replace(".","") for doc in documents]
processed_docs[2]


# In[37]:


# corpus is the collection of text
#look at the documents list
print("Our corpus: ", processed_docs)


# Initialise the object for CountVectorizer
count_vect = CountVectorizer()

#Build a BOW representation for the corpus
bow_rep = count_vect.fit_transform(processed_docs)

#Look at the vocabulary mapping
print("Our vocabulary: ", count_vect.vocabulary_)

#see the BOW rep for first 2 documents
print("BoW representation for 'dog bites man': ", bow_rep[0].toarray())
print("BoW representation for 'man bites dog: ",bow_rep[1].toarray())

#Get the representation using this vocabulary, for a new text
temp = count_vect.transform(["dog and dog are friends"])
print("Bow representation for 'dog and dog are friends':", temp.toarray())


# In[38]:


# Splitting x and y

X = data['text_clean'].values
y = data['Class'].values
y


# In[39]:


# Datatype for y is object. lets convert it into int
y = y.astype('int')
y


# In[41]:


type(X)


# In[42]:


## text preprocessing and feature vectorizer
# To extract features from a document of words, we import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


tf=TfidfVectorizer() ## object creation
X=tf.fit_transform(X) ## fitting and transforming the data into vectors


# In[43]:


X.shape


# In[44]:


## print feature names selected from the raw documents
len(tf.get_feature_names_out())


# In[46]:


X


# In[47]:


## getting the feature vectors
X=X.toarray()


# In[48]:


## Creating training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=6)


# In[49]:


## Model creation
from sklearn.naive_bayes import BernoulliNB

## model object creation
nb=BernoulliNB(alpha=0.01) 

## fitting the model
nb.fit(X_train,y_train)

## getting the prediction
y_hat=nb.predict(X_test) 


# In[50]:


y_hat


# In[51]:


## Evaluating the model
from sklearn.metrics import classification_report,confusion_matrix


# In[52]:


print(classification_report(y_test,y_hat))


# In[53]:


accuracyScore = accuracy_score(y_test,y_hat)*100
print("Predication accuracy:",accuracyScore)


# In[54]:


## confusion matrix
pd.crosstab(y_test,y_hat)


# In[ ]:




