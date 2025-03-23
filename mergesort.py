import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load Dataset
df = pd.read_csv('mobile.csv')

# Handle Missing Values
df.fillna(df.median(numeric_only=True), inplace=True)  # Fill numeric columns with median
df.fillna("Unknown", inplace=True)  # Fill categorical columns with "Unknown"

# Identify Categorical Columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Convert Categorical Columns using Label Encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Convert text to numeric
    label_encoders[col] = le  # Save encoders for future use

# Separate Features and Target
X = df.drop(columns=['price_range'])  # Independent Variables
y = df['price_range']  # Target Variable

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict on Test Data
y_pred = model.predict(X_test_scaled)

# Convert Predictions to DataFrame for Plotting
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Scatter Plot: Actual vs Predicted Prices
# plt.figure(figsize=(8, 5))
# sns.scatterplot(x=y_test, y=y_pred)
# plt.xlabel("Actual Price Range")
# plt.ylabel("Predicted Price Range")
# plt.title("Actual vs Predicted Price Range (Linear Regression)")
# plt.show()

# Print Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")