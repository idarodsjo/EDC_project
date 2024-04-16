from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#------ A great resource can be found here: https://developer.ibm.com/tutorials/awb-implementing-linear-discriminant-analysis-python/ 


# Define column names
cls = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']


# Read the data set
dataset = pd.read_csv("EDC_project/Iris_task/data_file.txt", names=cls)


# Dividing data set into features (X) and target variable (y)
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values


# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)


# Creating pair plot to illustrate realtionships between different features
# Diagonal elements show distribution of each feature
# Off-diagonal elements display scatterplots for each pair of features
ax = sns.pairplot(dataset, hue='Class', markers=["o", "s", "D"])
plt.suptitle("Pair Plot of Iris Dataset")
sns.move_legend(
    ax, "lower center",
    bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
plt.tight_layout()


# Creating histogram for visualising the distribution of individual features within the Iris data set
plt.figure(figsize = (12, 6))
for i, feature in enumerate(cls[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.histplot(data = dataset, x = feature, hue = 'Class', kde = True)
    plt.title(f'{feature} Distribution')

plt.tight_layout()


# Correlation heatmap offers insight into relationships between different features in the data set
correlation_matrix = dataset.corr(numeric_only = True)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")

# Splitting the data set into the training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# Implementing Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)


# Visualising data to understand the separability of the classes using scatterplot
tmp_Df = pd.DataFrame(X_train, columns=['LDA Component 1','LDA Component 2'])
tmp_Df['Class'] = y_train

sns.FacetGrid(tmp_Df, hue = "Class",
              height = 6).map(plt.scatter,
                              'LDA Component 1',
                              'LDA Component 2')

plt.legend(loc='upper right')


# Classify the data with random forest
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# Evaluating LDA model using accuracy score
accuracy = accuracy_score(y_test, y_pred)
conf_m = confusion_matrix(y_test, y_pred)


#Display the accuracy
print(f'Accuracy: {accuracy:.2f}')


#Display the confusion matrix as a heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(conf_m, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ---- Another great resource: https://www.geeksforgeeks.org/ml-linear-discriminant-analysis/
# --------- Goes more into detail about the math of it all