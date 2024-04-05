#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud
import re
import string
import nltk
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')

warnings.filterwarnings("ignore")

plt.style.use("seaborn")


# In[4]:


train=pd.read_csv("train_E6oV3lV.csv")


# In[5]:


test=pd.read_csv("test_tweets_anuFYb8.csv")


# In[6]:


train.shape, test.shape


# In[7]:


train.head()


# In[8]:


test.head()


# In[9]:


length_train = train['tweet'].str.len()
length_test = test['tweet'].str.len()
plt.figure(figsize=(16,6))
plt.hist(length_train , bins = 50 , label ="Train_Tweets",color = "skyblue")
plt.hist(length_test, bins = 50 , label = "Test_Tweets")
plt.legend()


# In[10]:


df = train
df.shape


# In[11]:


df


# In[12]:


def remove_pattern(input_text,pattern):
    r = re.findall(pattern,input_text)
    
    for word in r:
        input_text = re.sub(word,"",input_text)
    
    return input_text


# In[13]:


df['clean_tweet'] = np.vectorize(remove_pattern)(df['tweet'],"@[\w]*")


# In[14]:


df


# In[16]:


df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]"," ")
df


# In[17]:


df['clean_tweet'] = df['clean_tweet'].apply(lambda x : " ".join([word for word in x.split() if len(word)>3]))
df.head()


# In[18]:


tokenized_tweet = df['clean_tweet'].apply(lambda x : x.split())
tokenized_tweet.head()


# In[19]:


from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda sentence : [stemmer.stem(word) for word in sentence])
tokenized_tweet.head()


# In[20]:


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = " ".join(tokenized_tweet[i])

tokenized_tweet.head()


# In[21]:


df['clean_tweet'] = tokenized_tweet
df.head()


# In[22]:


all_words = " ".join([sentence for sentence in df['clean_tweet']]) 


# In[23]:


wordcloud = WordCloud(width = 800 , height = 500 , random_state = 42 , max_font_size = 100).generate(all_words)


# In[24]:


plt.figure(figsize = (16,8))
plt.imshow(wordcloud , interpolation = "bilinear")
plt.axis("off")
plt.show()


# In[26]:


all_words = " ".join([sentence for sentence in df['clean_tweet'][df['label']==0]])

wordcloud = WordCloud(width = 800 , height = 500 , random_state = 42 , max_font_size = 100).generate(all_words)

plt.figure(figsize = (16,8))
plt.imshow(wordcloud , interpolation = "bilinear")
plt.axis("off")
plt.show()


# In[27]:


all_words = " ".join([sentence for sentence in df['clean_tweet'][df['label']==1]]) 

wordcloud = WordCloud(width = 800 , height = 500 , random_state = 42 , max_font_size = 100).generate(all_words)

plt.figure(figsize = (16,8))
plt.imshow(wordcloud , interpolation = "bilinear")
plt.axis("off")
plt.show()


# In[28]:


def add_value_labels(ax, spacing=5):
    
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        
        space = spacing
        va = 'bottom'
        
        if y_value < 0:
            space *= -1
            va = 'top'
            
        label = "{:.1f}".format(y_value)
        
        ax.annotate(label,(x_value, y_value),xytext=(0, space),textcoords="offset points",ha='center',va=va)   


# In[29]:


def hashtag_extract(tweets):
    hashtag = []
    for tweet in tweets:
        ht = re.findall(r"#(\w+)",tweet)
        hashtag.append(ht)
    return hashtag


# In[30]:


ht_positive = hashtag_extract(df['clean_tweet'][df['label']==0])

ht_negative = hashtag_extract(df['clean_tweet'][df['label']==1])


# In[31]:


ht_positive[:5]


# In[32]:


ht_positive = sum(ht_positive,[])
ht_negative = sum(ht_negative,[])
ht_positive[:5]


# In[33]:


freq = nltk.FreqDist(ht_positive)

d = pd.DataFrame({'Hashtag':list(freq.keys()),'Count':list(freq.values())})

d.head()


# In[34]:


d = d.nlargest(columns='Count',n=10)

plt.figure(figsize=(16,6))
ax = sns.barplot(data=d,x="Hashtag",y="Count")
plt.title("Top 10 Positive Hashtag Words",fontweight="bold",fontsize=25)
add_value_labels(ax)
plt.show()


# In[35]:


freq = nltk.FreqDist(ht_negative)

d = pd.DataFrame({'Hashtag':list(freq.keys()),'Count':list(freq.values())})

d.head()


# In[36]:


d = d.nlargest(columns='Count',n=10)

plt.figure(figsize=(16,6))
ax = sns.barplot(data=d,x="Hashtag",y="Count")
plt.title("Top 10 Negative Hashtag Words",fontweight="bold",fontsize=25)
add_value_labels(ax)
plt.show()


# In[37]:


from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_df = .90 , min_df = 2 , max_features = 1000 , stop_words = "english")
bow  = bow_vectorizer.fit_transform(df['clean_tweet'])


# In[38]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(bow,df['label'],random_state=42,test_size=.25)


# In[39]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score , accuracy_score , precision_score , recall_score,classification_report,confusion_matrix

model = LogisticRegression()
model.fit(x_train,y_train)
pred = model.predict(x_test)

accuracy = accuracy_score(y_test,pred)
precision = precision_score(y_test,pred)
recall = recall_score(y_test,pred)
f1 = f1_score(y_test,pred)
report = classification_report(y_test,pred)
matrix = confusion_matrix(y_test,pred)

print('Accuracy: {:.2f}'.format(accuracy))
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1Score: {:.2f}'.format(f1))
print('classification report:\n',report)
print('confusion matrix:\n',matrix)


# In[41]:


pred_prob = model.predict_proba(x_test)

pred = pred_prob[:,1] >=0.3
pred = pred.astype(int)

accuracy = accuracy_score(y_test,pred)
precision = precision_score(y_test,pred)
recall = recall_score(y_test,pred)
f1 = f1_score(y_test,pred)
report = classification_report(y_test,pred)
matrix = confusion_matrix(y_test,pred)

print('Accuracy: {:.2f}'.format(accuracy))
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1Score: {:.2f}'.format(f1))
print('classification report:\n',report)
print('confusion matrix:\n',matrix)


# In[ ]:




