# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sklearn
import sklearn as sk
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import svm
from six import StringIO


from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score



df = pd.read_csv("train.csv", )
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

df['Fare'] = df['Fare'].fillna(df['Fare'].dropna().median())
df['Embarked'] = df['Embarked'].fillna('S')
df['Age']=df['Age'].fillna('0')
df.loc[df['Sex']=='male','Sex']=0
df.loc[df['Sex']=='female','Sex']=1
df.loc[df['Embarked']=='S','Embarked']=0
df.loc[df['Embarked']=='C','Embarked']=1
df.loc[df['Embarked']=='Q','Embarked']=2
exclude = ['Name','Cabin','Ticket']
df = df.drop(exclude, axis=1)
#print(df.head())

X = df.drop("Survived",axis=1)
y = df["Survived"]

mdlsel = SelectKBest(chi2, k=4)
mdlsel.fit(X,y)
i = mdlsel.get_support()
data2 = pd.DataFrame(mdlsel.transform(X), columns = X.columns.values[i])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

scores = cross_val_score(clf, X, y, cv=5)

print("Linear accuracy = %0.2f" % (scores.mean()))

clf1 = svm.SVC(kernel='poly', degree=2)
clf1.fit(X_train, y_train)

y_pred1 = clf1.predict(X_test)

scores1 = cross_val_score(clf1, X, y, cv=5)

print("Polynomial accuracy = %0.2f" % (scores1.mean()))

svclassifier = svm.SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)

y_pred2 = svclassifier.predict(X_test)

scores2 = cross_val_score(svclassifier, X, y, cv=5)

print("RBF accuracy = %0.2f" % (scores2.mean()))





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
