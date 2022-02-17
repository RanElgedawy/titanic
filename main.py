# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sklearn
import sklearn as sk
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn import tree , ensemble
from sklearn.ensemble import RandomForestRegressor


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
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth = 4)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)



#Predict the response for test dataset
y_pred = clf.predict(X_test)

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = X.columns,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('titanic.png')
Image(graph.create_png())

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cnt = 1
# split()  method generate indices to split data into training and test set.
for train_index, test_index in kf.split(X, y):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt+=1

score = cross_val_score(tree.DecisionTreeClassifier(random_state= 42), X, y, cv= kf, scoring="accuracy")
print(f'Scores for each fold are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train)

predictions = rf.predict(X_test)

score = cross_val_score(ensemble.RandomForestClassifier(random_state= 42), X, y, cv= kf, scoring="accuracy")
print(f'Scores for each fold are: {score}')
print(f'Average score: {"{:.2f}".format(score.mean())}')




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
