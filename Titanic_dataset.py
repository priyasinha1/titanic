#!/usr/bin/env python
# coding: utf-8

# Titanic Dataset

# In[126]:


import pandas as pd

train = pd.read_csv('titanic_train.csv')
test =pd.read_csv('titanic_test.csv')


# In[127]:


train.head(10)


# In[128]:


test.head(10)


# In[129]:


test.info()


# In[130]:


train.info()


# In[131]:


train.isnull().sum()


# In[132]:


test.isnull().sum()


# In[133]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[134]:


def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize =(10,5))
    


# In[135]:


bar_chart('Sex')


# In[136]:


bar_chart('Pclass')


# In[137]:


bar_chart('SibSp')


# In[138]:


bar_chart('Parch')


# In[139]:


bar_chart('Embarked')


# In[140]:


train_test_data = [train, test]
for dataset in train_test_data:
    dataset['Title']= dataset['Name'].str.extract('([A-Za-z]+)\.',expand = False)


# In[141]:


train['Title'].value_counts()


# In[142]:


test['Title'].value_counts()


# In[ ]:





# In[143]:


title_mapping = {"Mr":0,"Miss":1,"Mrs": 2,"Master":3,"Dr":3,"Rev":3,"Major":3,"Col":3,"Mlle":3,"Mme":3,"Jonkheer":3,"Capt":3,"Countess":3,"Ms":3,"Sir":3,"Lady":3,"Don":3}
for dataset in train_test_data:
    dataset['Title']= dataset['Title'].map(title_mapping)                        


# In[144]:


train.head()


# In[145]:


test.head()


# In[146]:


bar_chart('Title')


# In[147]:


train.drop('Name',axis=1,inplace = True)


# In[148]:


train.head()


# In[149]:


test.drop('Name',axis=1,inplace= True)


# In[150]:


test.head()


# In[151]:


sex_mapping = {'male':0,'female':1}
for dataset in train_test_data:
    dataset['Sex']= dataset['Sex'].map(sex_mapping) 


# In[152]:


train.head()


# In[153]:


test.head()


# In[154]:


bar_chart('Sex')


# In[155]:


train['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace = True)
test['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace = True)


# In[156]:


facet = sns.FacetGrid(train, hue='Survived',aspect = 4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim = (0, train ['Age'].max()))
facet.add_legend()
plt.show()


# In[157]:


for dataset in train_test_data:
    dataset.loc[dataset['Age'] <= 16, 'Age']=0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age']<= 26),'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age']<= 36),'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age']<= 62),'Age'] = 3,
    dataset.loc[dataset['Age'] > 62,'Age']=4


# In[158]:


train.head()


# In[159]:


bar_chart('Age')


# In[160]:


Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index = ['1st class','2nd class','3rd class']
df.plot(kind='bar',stacked=True, figsize =(10,5))


# In[161]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna("S")


# In[162]:


train.head()


# In[163]:


embarked_mapping= {'S':0,'C':1,'Q':2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# In[164]:


train.head()


# In[165]:


test.head()


# In[166]:


bar_chart('Embarked')


# In[167]:


train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform('median'),inplace = True)
test["Fare"].fillna(train.groupby("Pclass")["Fare"].transform('median'),inplace = True)


# In[168]:


facet = sns.FacetGrid(train, hue='Survived',aspect = 4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim = (0, train ['Fare'].max()))
facet.add_legend()
plt.show()


# In[169]:


for dataset in train_test_data:
    dataset.loc[dataset['Fare'] <= 17, 'Fare']=0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare']<= 32),'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 32) & (dataset['Fare']<= 100),'Fare'] = 2,
    dataset.loc[dataset['Fare'] > 100,'Fare']=3


# In[170]:


train.head()


# In[171]:


test.head()


# In[172]:


train.Cabin.value_counts()


# In[173]:


for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# In[174]:


Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index = ['1st class','2nd class','3rd class']
df.plot(kind='bar',stacked=True, figsize =(10,5))


# In[175]:


cabin_mapping = {'A':0,'B':0.4,'C':0.8,'D':1.2,'E':1.6,'F':2.0,'G':2.4,'T':2.8}
for dataset in train_test_data:
    dataset['Cabin']=dataset['Cabin'].map(cabin_mapping)


# In[176]:


train.head()


# In[177]:


train['Cabin'].fillna(train.groupby("Pclass")["Cabin"].transform("median"),inplace = True)
test['Cabin'].fillna(train.groupby("Pclass")["Cabin"].transform("median"),inplace = True)


# In[178]:


train.head()


# In[179]:


train['FamilySize'] = train["SibSp"] + train["Parch"]+1
test['FamilySize'] = test["SibSp"] + test["Parch"]+1


# In[181]:


facet = sns.FacetGrid(train, hue='Survived',aspect = 4)
facet.map(sns.kdeplot,'FamilySize',shade= True)
facet.set(xlim = (0, train ['FamilySize'].max()))
facet.add_legend()
plt.show()


# In[182]:


family_mapping = {1:0, 2:0.4, 3:0.8, 4:1.2, 5:1.6, 6:2.0, 7:2.4, 8:2.8, 9:3.2, 10:3.6, 11:4.0}
for dataset in train_test_data:
    dataset["FamilySize"] = dataset["FamilySize"].map(family_mapping)


# In[183]:


train.head()


# In[184]:


feature_drop = ['Ticket','SibSp','Parch']
train = train.drop(feature_drop, axis = 1)
test = test.drop(feature_drop, axis = 1)
train= train.drop('PassengerId',axis = 1)


# In[185]:


train.head()


# In[186]:


test.head()


# In[187]:


train_data = train.drop('Survived',axis = 1)
target = train['Survived']
train_data.shape, target.shape


# In[188]:


train_data.head()


# In[189]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np


# In[190]:


train.info()


# In[191]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits = 10, shuffle = True, random_state=0)


# In[192]:


clf= KNeighborsClassifier(n_neighbors =13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[193]:


round(np.mean(score)*100,2)


# In[194]:


clf= DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[195]:


round(np.mean(score)*100,2)


# In[196]:


clf= RandomForestClassifier(n_estimators =13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[197]:


round(np.mean(score)*100,2)


# In[198]:


clf= GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[199]:


round(np.mean(score)*100,2)


# In[200]:


clf= SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[201]:


round(np.mean(score)*100,2)


# In[202]:


clf =SVC()
clf.fit(train_data, target)

test_data = test.drop("PassengerId", axis = 1).copy()
prediction = clf.predict(test_data)


# In[123]:


submission = pd.DataFrame({
    "passengerId":test["PassengerId"],
    "Survived": prediction
})

submission.to_csv('submission.csv',index=False)


# In[125]:


submission = pd.read_csv('submission.csv')


# In[204]:


train_data.head


# In[205]:


train_data['Cabin'].fillna(train_data.groupby("Pclass")["Cabin"].transform("median"),inplace = True)


# In[206]:


clf =SVC()
clf.fit(train_data, target)

test_data = test.drop("PassengerId", axis = 1).copy()
prediction = clf.predict(test_data)


# In[210]:


test_data.head(300)


# In[211]:


test_data['Cabin'].fillna(test_data.groupby("Pclass")["Cabin"].transform("median"),inplace = True)


# In[212]:


clf =SVC()
clf.fit(train_data, target)

test_data = test.drop("PassengerId", axis = 1).copy()
prediction = clf.predict(test_data)


# In[ ]:




