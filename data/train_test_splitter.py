import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load the JSON Data
data = pd.read_json('data.json')

# Step 2: Define Features and Target
X = data.iloc[:, :-1]  # Features (all columns except the target)
y = data.iloc[:, -1]   # Target variable (the last column)

# Combine features and target into a single DataFrame for the split
combined_data = pd.concat([X, y], axis=1)

# Step 3: Split the Data into Training and Test Sets
train_data, test_data = train_test_split(combined_data, test_size=0.2, random_state=42)

# Step 4: Save the Training and Test Sets
train_data.to_csv('train_data.csv', index=False)  # Save training set
test_data.to_csv('test_data.csv', index=False)    # Save test set

# Print confirmation
print("Train and test sets saved as 'train_data.csv' and 'test_data.csv'.")
