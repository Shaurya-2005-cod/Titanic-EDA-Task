# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
data = pd.read_csv('MultipleFiles/Titanic-Dataset.csv')  # Adjust the path if necessary

# 1. Generate Summary Statistics
summary_statistics = data.describe(include='all')  # Include all columns for summary
print("Summary Statistics:")
print(summary_statistics)

# 2. Create Histograms for Numeric Features
numeric_features = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_features].hist(bins=15, figsize=(15, 10))
plt.suptitle('Histograms of Numeric Features')
plt.show()

# 3. Create Boxplots for Numeric Features
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numeric_features):
    plt.subplot(3, 3, i + 1)  # Adjust the number of rows and columns as needed
    sns.boxplot(y=data[feature])
    plt.title(feature)
plt.tight_layout()
plt.show()

# 4. Use Pairplot for Feature Relationships
sns.pairplot(data[numeric_features])
plt.suptitle('Pairplot of Numeric Features', y=1.02)
plt.show()

# 5. Generate Correlation Matrix
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# 6. Identify Patterns, Trends, or Anomalies
# Example: Scatter plot for Age vs Fare
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Age'], y=data['Fare'], hue=data['Survived'], alpha=0.6)
plt.title('Scatter Plot of Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend(title='Survived', loc='upper left', labels=['No', 'Yes'])
plt.show()

# 7. Save the summary statistics to a CSV file (optional)
summary_statistics.to_csv('summary_statistics.csv')

# Pause the terminal
input("Press Enter to continue...")
