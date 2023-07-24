import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

df = pd.read_csv('sleep.csv')
df = df.iloc[:, 1:] #remove first column: Person ID
# print(df.isnull().sum()) #check for missing values
df.fillna('None', inplace=True) #replace None values as strings
# print(df['Sleep Disorder'].value_counts())
# print(df.shape()) # 374 x 13
# print(df.info())

# predictors_list = list(X.columns)

X = df.drop(['Sleep Disorder'], axis=1)
y = df['Sleep Disorder']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# print(X_train.shape, X_test.shape)

encoder = ce.OrdinalEncoder(cols=['Gender', 'Occupation', 'BMI Category', 'Blood Pressure'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(accuracy_score(y_test, y_pred))

rfc_10 = RandomForestClassifier(random_state=0, n_estimators=10)
rfc_10.fit(X_train, y_train)
y_pred_10 = rfc_10.predict(X_test)
# print(accuracy_score(y_test, y_pred_10))

feature_scores = pd.Series(rfc.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(feature_scores)
# sns.barplot(x=feature_scores, y=feature_scores.index)
# plt.xlabel('Feature Importance Score')
# plt.ylabel('Features')
# plt.title('Visualising Important Features')
# plt.show()

# print(df.head())
new_X = df.drop(['Sleep Disorder', 'Sleep Duration'], axis=1)
# print(new_X.head())
new_y = df['Sleep Disorder']
new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(new_X, new_y, test_size=0.33, random_state=42)
print(new_X_train.shape, new_X_test.shape, new_y_train.shape, new_y_test.shape)
new_encoder = ce.OrdinalEncoder(cols=['Gender', 'Occupation', 'BMI Category', 'Blood Pressure'])
new_X_train = new_encoder.fit_transform(new_X_train)
new_X_test = new_encoder.transform(new_X_test)
new_rfc = RandomForestClassifier(random_state=0)
new_rfc.fit(new_X_train, new_y_train)
new_y_pred = new_rfc.predict(new_X_test)
print(accuracy_score(new_y_test, new_y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=rfc.classes_).plot()
# plt.show()
print(classification_report(y_test, y_pred))