# Import the libraries
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
# import joblib


# Load the dataset
df = pd.read_csv(r"E:\projects\health_app\data\diabetes.csv")

# Splitting dependent & independent variables
y = df['Outcome']
x = df.drop(df[['Outcome']], axis=1)

# Splitting into train & test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

# Logistic Regression Model
model = LogisticRegression()

# Training the model
model = model.fit(X_train, y_train)

# Predicting the y values
y_pred = model.predict(X_test)

# Printing accuracy score
print("Accuracy: ", accuracy_score(y_test, y_pred)*100)

# Classification Report
clf_report = classification_report(y_test, y_pred)
print('Classification report')
print("---------------------")
print(clf_report)
print("_____________________")

# Confusion Matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print("---------------------")
print(cnf_matrix)
print("_____________________")

# joblib.dump(model, r"E:\projects\health\model\diabetes_model.pkl")
pickle.dump(model, open('diabetes_model.pkl', 'wb'))
