import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pickle
train_df = pd.read_csv('FullDataSet/Train.csv')
test_df = pd.read_csv('FullDataSet/Test.csv')
#Removing StopWords
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()

sentences = []
# count=0
for sent in train_df['Text']:
#     print(count)
    sent = re.sub('[^a-zA-Z]', ' ',sent)
    sent = sent.lower()
    sent = sent.split()
    sent = [ lemmatizer.lemmatize(word) for word in sent if word not in set(stopwords.words('english')) ]
    sent = ' '.join(sent)
    sentences.append(sent)
    print("T")
#     count+=1
train_df['new_text'] = sentences

sentences = []
# count=0
for sent in test_df['Text']:
#     print(count)
    sent = re.sub('[^a-zA-Z]', ' ',sent)
    sent = sent.lower()
    sent = sent.split()
    sent = [ lemmatizer.lemmatize(word) for word in sent if word not in set(stopwords.words('english')) ]
    sent = ' '.join(sent)
    sentences.append(sent)
    print("D")
#     count+=1
test_df['new_text'] = sentences
train_df['Target'] = train_df['Target'].map({'angry': 0 ,'sad':1, 'relaxed':2 , 'happy': 3} )

test_df['Target'] = test_df['Target'].map({'angry': 0 ,'sad':1, 'relaxed':2 , 'happy': 3} )
print("d")
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pickle
tf = TfidfVectorizer(max_features=5000)
df1 = tf.fit_transform(train_df['Text']).toarray()
pickle.dump(tf,open("feature.pkl","wb"))

df2 = tf.transform(test_df['Text']).toarray()
df1 = pd.DataFrame(df1)
df1 = pd.concat([train_df['Target'],df1],axis=1)
df2 = pd.DataFrame(df2)
df2 = pd.concat([test_df['Target'],df2],axis=1)


from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()

model.fit(df1.drop("Target",axis=1),df1['Target'])

y_predicted = model.predict(df2.drop("Target",axis=1))

confusion_matrix(df2['Target'],y_predicted)

accuracy_score(df2['Target'],y_predicted)



from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(df1.drop("Target",axis=1),df1['Target'])

y_predicted = model.predict(df2.drop("Target",axis=1))

confusion_matrix(df2['Target'],y_predicted)

accuracy_score(df2['Target'],y_predicted)



from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(df1.drop("Target",axis=1),df1['Target'])

y_predicted = model.predict(df2.drop("Target",axis=1))

confusion_matrix(df2['Target'],y_predicted)

accuracy_score(df2['Target'],y_predicted)



from sklearn.svm import SVC
model = SVC()

model.fit(df1.drop("Target",axis=1),df1['Target'])

y_predicted = model.predict(df2.drop("Target",axis=1))

confusion_matrix(df2['Target'],y_predicted)

accuracy_score(df2['Target'],y_predicted)



#DumpingOut Trained Models


f4 = 'SVC.sav'

filehandler = open(f4,"wb")
pickle.dump(model,filehandler)
filehandler.close()

