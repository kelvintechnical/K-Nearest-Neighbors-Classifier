
# <h1>K-Nearest Neighbors (KNN) Classifier</h1>

<p>Welcome to the <strong>K-Nearest Neighbors (KNN) Classifier</strong> project! This repository contains code to implement a basic machine learning classifier using the K-Nearest Neighbors algorithm. This algorithm classifies data points based on the majority class of the closest data points, or "neighbors."</p>

---

## 📫 How to reach me:

**Email**: [ktobia10@wgu.edu](mailto:ktobia10@wgu.edu)  
**LinkedIn**: [Kelvin R. Tobias](https://www.linkedin.com/in/kelvin-r-tobias-211949219/)  
**Bluesky**: [@kelvintechnical.bsky.social](https://bsky.app/profile/kelvintechnical.bsky.social)  
**Instagram**: [@kelvinintech](https://www.instagram.com/kelvinintech/)  

---

## <h2>Project Overview</h2>

<p>The <strong>K-Nearest Neighbors (KNN) Classifier</strong> is a simple and effective classification algorithm that works well with smaller datasets and is easy to understand. In this project, we use Python libraries like <code>numpy</code>, <code>matplotlib</code>, and <code>scikit-learn</code> to build and test our KNN model on the popular Iris dataset, classifying different species of iris flowers based on physical features.</p>

---

## <h2>5 Things I Learned from This Project</h2>
<p>While implementing and reviewing the code, here are five key insights I gained:</p>
<ol>
  <li><strong>Importance of Data Splitting:</strong> Splitting data into training and testing sets helps prevent overfitting and ensures the model generalizes well to unseen data.</li>
  <li><strong>Cross-Validation:</strong> Using cross-validation provides a better estimate of model performance by evaluating it on multiple subsets of the dataset.</li>
  <li><strong>Feature Extraction:</strong> The <code>CountVectorizer</code> transforms text into numerical data by counting word occurrences, which is crucial for text-based machine learning.</li>
  <li><strong>Saving Models:</strong> The <code>joblib</code> library allows for saving and loading models and preprocessing tools, making it easier to deploy and reuse them.</li>
  <li><strong>Understanding Metrics:</strong> Metrics like precision, recall, and F1-score provide a detailed view of model performance beyond just accuracy.</li>
</ol>

---

## <h2>Code Explanation</h2>

<p>Here is an overview of the key components of the code in this project:</p>

```python
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Test the model
accuracy = knn.score(X_test, y_test)
print("Model accuracy:", accuracy)
```

<h3>Explanation:</h3>
<ul>
  <li><code>data = load_iris()</code>: Loads the Iris dataset, which contains data for classifying iris flowers.</li>
  <li><code>train_test_split</code>: Splits data into training and testing sets.</li>
  <li><code>KNeighborsClassifier</code>: Initializes the KNN classifier with <code>n_neighbors=3</code>.</li>
</ul>

---

## <h2>How to Use This Repository</h2>

<p>If you want to use or modify this code, you can "fork" it to make your own copy:</p>

<ol>
  <li>Fork this repository by clicking the "Fork" button at the top-right of this page.</li>
  <li>Clone the forked repository to your local machine:</li>
</ol>

```bash
git clone https://github.com/your-username/K-Nearest-Neighbors-KNN-Classifier.git
```

<ol start="3">
  <li>Navigate to the project directory:</li>
</ol>

```bash
cd K-Nearest-Neighbors-KNN-Classifier
```

<ol start="4">
  <li>Install the necessary libraries:</li>
</ol>

```bash
pip install numpy scikit-learn matplotlib
```

<ol start="5">
  <li>Run the code:</li>
</ol>

```bash
python knn_classifier.py
```

---

## <h2>Contributing</h2>

<p>Contributions are welcome! Feel free to make pull requests to improve the code or add new features.</p>

---

## <h2>License</h2>

<p>This project is open-source and free to use. Please credit this repository if you use it in your own projects.</p>
