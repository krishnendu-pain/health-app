# Import the libraries
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


# Load the dataset
df = pd.read_csv(r"E:\projects\health\data\liver.csv")

df.rename(columns={'Dataset': 'disease'}, inplace=True)
df.dropna(inplace=True)

df['disease'] = df['disease'].replace(to_replace=2, value=0)
df['Gender'] = df['Gender'].replace(to_replace={'Male': 1, 'Female': 0})

# Splitting dependent & independent variables
y = df['disease']
x = df.drop(df[['disease', 'Total_Bilirubin']], axis=1)

# Splitting into train & test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

# Logistic Regression Model
model = RandomForestClassifier()

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

joblib.dump(model, r"liver_model.pkl")
