from sklearn.feature_extraction.text import CountVectorizer #TURNS TEXT INTO NUMBERS // counting how often each word appears in the text
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import os
 #splits data into training and testing 
# to evaluate our model, we train it on one portion of the data, and test it on another
# The training data teaches the classifier how to recognize patterns, if we were a classifying emails as spam and not spam, the training data helps the model learn what spam looks like
#                               What happens if we don't split the data?
# The model might memorize the training data instead of learning general patterns. Called over fitting.
from sklearn.naive_bayes import MultinomialNB #use dwhen our data represents counts (like word frequencies in text)
from sklearn.metrics import accuracy_score #calculates how accurate our preditions are compared to the actual results
import pandas as pd
import joblib #provides tools to save and load Python objects
#accuracy_score takses two arguments (predition labels, true labels)
#predition lables: what the model thinks the labels are
#true labels: the actual correct labels


# Load the dataset (adjust the path if needed)
data = pd.read_csv('expanded_dataset.csv')  # If the file is in the same directory as the script

# or use an absolute path if the file is elsewhere:
# data = pd.read_csv(r"C:\Users\kelvi\OneDrive\Desktop\python projects\ML\Naive_Bayes_Text_Classifier\expanded_dataset.csv")

# Preview the dataset
print("Dataset loaded successfully!")
print(data.head())  # Print the first few rows to confirm structure

#transforms text into numerical features
vectorizer = CountVectorizer(stop_words='english', lowercase=True) #converts text into a maxtix of word counts 

#transform the 'text' column into the datasets into numnerical data

X = vectorizer.fit_transform(data['text']) #learn the vocabulary and encode the text

y = data['Labels'] #Assign the 'label' column to y 

# Define the classifier
classifier = MultinomialNB(alpha=0.5)  # Example


# Perform cross-validation
scores = cross_val_score(classifier, X, y, cv=2)  # 2-fold cross-validation

# Print scores for each fold
print("Cross-validation scores:", scores)

# Calculate and print mean accuracy
print(f"Mean cross-validation accuracy: {scores.mean() * 100:.2f}%")
##3333333333333333333333333333333333333333333333333


# Debugging: Print the columns of the DataFrame
print("Available columns in the DataFrame:")
print(data.columns)

# Debugging: Print the first few rows of the DataFrame
print("Preview of the data:")
print(data.head())



##333333333333333333333333333333333333333333333333

#Split the data into training and testing sets

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



#MultinomialNB classifier


#train the classifier using the training data
classifier.fit(X_train, y_train)  #fits model into the training data


y_pred = classifier.predict(X_test) #predicts labels for the testing set

accuracy = accuracy_score(y_test, y_pred) #compare predict labels with true labela 

print(f'Accruacy: {accuracy * 100:.2f}%') #convert accruacy to a percentage
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


#Create new messages to classify

new_messages = [
    "Congratulations, you won a free ticket!",
    "Don't forget about our meeting tomorrow.",
    "Claim your prize by clicking this link now.",
    "how was your day today?"
]

#Transform the messages into numerical data
new_messages_transformed = vectorizer.transform(new_messages)   #use the same vectorizer we trained earlier 

predictions = classifier.predict(new_messages_transformed)

for message, label in zip(new_messages, predictions):
    print(f"Message: '{message}' is classified as: {label}")


#IMPORT JOBLIB TO DEPLOY IN REALWORLD APPLICATIONS 
joblib.dump(classifier, "naive_bayes_classifier.pkl") #saves model as a pkl file

joblib.dump(vectorizer, "count_vectorizer.pkl")  #save the vectorizer for preprocessing new data

print("Model and vectorizer saved successfully!")

#load the saved model and vectorizer
loaded_classifer = joblib.load("naive_bayes_classifier.pkl")
loaded_vectorizer = joblib.load("count_vectorizer.pkl")

#simple loop for user input

while True:
    message = input("Enter a message to classify (or 'quit' to stop): ")
    if message.lower() == 'quit':
        break

    new_messages_transformed = loaded_vectorizer.transform([message])

    prediction = loaded_classifer.predict(new_messages_transformed)

    print(f"The message is classified as: {prediction[0]}")