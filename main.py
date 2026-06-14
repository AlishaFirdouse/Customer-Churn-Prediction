import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Remove customerID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df.fillna(0, inplace=True)

# Encode all text columns
from sklearn.preprocessing import LabelEncoder

for col in df.columns:
    if df[col].dtype != "int64" and df[col].dtype != "float64":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
print("\nAfter Encoding:\n")
print(df.head())
print("\n")
print(df.dtypes)

# Features and Target
print("\nColumns with object datatype:")
print(df.select_dtypes(include=['object']).columns)



X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Customer Churn Confusion Matrix")
plt.show()

# ==============================
# FEATURE IMPORTANCE
# ==============================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

feature_importance = feature_importance.sort_values(
    by="Coefficient",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)


# ==============================
# CHURN PROBABILITY
# ==============================

print("\nCustomer Churn Probabilities (First 10 Customers):")

probabilities = model.predict_proba(X_test)

for i in range(10):
    churn_probability = probabilities[i][1] * 100

    if churn_probability > 80:
        risk = "High Risk"
    elif churn_probability > 50:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    print(
        f"Customer {i+1}: "
        f"Churn Probability = {churn_probability:.2f}% "
        f"-> {risk}"
    )


# ==============================
# MODEL COMPARISON
# ==============================

print("\nModel Comparison")

# Logistic Regression
lr_accuracy = accuracy_score(y_test, y_pred)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"Logistic Regression Accuracy : {lr_accuracy*100:.2f}%")
print(f"Decision Tree Accuracy       : {dt_accuracy*100:.2f}%")
print(f"Random Forest Accuracy       : {rf_accuracy*100:.2f}%")

best_accuracy = max(
    lr_accuracy,
    dt_accuracy,
    rf_accuracy
)

if best_accuracy == lr_accuracy:
    print("\nBest Model : Logistic Regression")
elif best_accuracy == dt_accuracy:
    print("\nBest Model : Decision Tree")
else:
    print("\nBest Model : Random Forest")

# ==============================
# MANUAL CUSTOMER PREDICTION
# ==============================

print("\nEnter New Customer Details")

gender = int(input("Gender (0=Female, 1=Male): "))
senior = int(input("Senior Citizen (0=No, 1=Yes): "))
partner = int(input("Partner (0=No, 1=Yes): "))
dependents = int(input("Dependents (0=No, 1=Yes): "))
tenure = int(input("Tenure: "))

# Default values for remaining features
#new_customer = [[
   # gender,
    #senior,
    #partner,
   # dependents,
   # tenure,
   # 1,  # PhoneService
   # 1,  # MultipleLines
   # 1,  # InternetService
   # 1,  # OnlineSecurity
   # 1,  # OnlineBackup
   # 1,  # DeviceProtection
   # 1,  # TechSupport
   # 1,  # StreamingTV
    #1,  # StreamingMovies
   # 1,  # Contract
    #1,  # PaperlessBilling
   # 1,  # PaymentMethod
    #50.0,   # MonthlyCharges
   # 1000.0  # TotalCharges]]
new_customer = pd.DataFrame([[
    gender,
    senior,
    partner,
    dependents,
    tenure,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1,
    70.5,
    1000
]], columns=X.columns)

prediction = model.predict(new_customer)
probability = model.predict_proba(new_customer)

print("\nPrediction Result")

if prediction[0] == 1:
    print("Customer is likely to CHURN")
else:
    print("Customer is likely to STAY")

print(f"Churn Probability: {probability[0][1]*100:.2f}%")

pickle.dump(model, open("churn_model.pkl", "wb"))

print("Model saved successfully!")