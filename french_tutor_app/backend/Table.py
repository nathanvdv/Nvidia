import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from prettytable import PrettyTable

df = pd.read_csv('data/training_data.csv')
X_train, X_test, y_train, y_test = train_test_split(df['sentence'], df['difficulty'], test_size=0.2, random_state=42)

logreg_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression(max_iter=1000))
])

knn_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', KNeighborsClassifier())
])

dt_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', DecisionTreeClassifier())
])

rf_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', RandomForestClassifier())
])

svm_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', SVC())
])

param_grid_logreg = {
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'classifier__C': [0.1, 1, 10]
}

param_grid_knn = {
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'classifier__n_neighbors': [3, 5, 7]
}

param_grid_dt = {
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'classifier__max_depth': [None, 10, 20]
}

param_grid_rf = {
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'classifier__n_estimators': [50, 100, 200]
}

param_grid_svm = {
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf']
}

grid_search_logreg = GridSearchCV(logreg_pipeline, param_grid_logreg, cv=5, scoring='accuracy')
grid_search_knn = GridSearchCV(knn_pipeline, param_grid_knn, cv=5, scoring='accuracy')
grid_search_dt = GridSearchCV(dt_pipeline, param_grid_dt, cv=5, scoring='accuracy')
grid_search_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=5, scoring='accuracy')
grid_search_svm = GridSearchCV(svm_pipeline, param_grid_svm, cv=5, scoring='accuracy')

grid_search_logreg.fit(X_train, y_train)
grid_search_knn.fit(X_train, y_train)
grid_search_dt.fit(X_train, y_train)
grid_search_rf.fit(X_train, y_train)
grid_search_svm.fit(X_train, y_train)

logreg_predictions = grid_search_logreg.predict(X_test)
knn_predictions = grid_search_knn.predict(X_test)
dt_predictions = grid_search_dt.predict(X_test)
rf_predictions = grid_search_rf.predict(X_test)
svm_predictions = grid_search_svm.predict(X_test)

print("Logistic Regression Classification Report:")
print(classification_report(y_test, logreg_predictions))

print("\nkNN Classification Report:")
print(classification_report(y_test, knn_predictions))

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_predictions))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_predictions))

print("\nSVM Classification Report:")
print(classification_report(y_test, svm_predictions))

logreg_accuracy = accuracy_score(y_test, logreg_predictions)
knn_accuracy = accuracy_score(y_test, knn_predictions)
dt_accuracy = accuracy_score(y_test, dt_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)

print("\nAccuracy for each model:")
print(f"Logistic Regression: {logreg_accuracy:.4f}")
print(f"kNN: {knn_accuracy:.4f}")
print(f"Decision Tree: {dt_accuracy:.4f}")
print(f"Random Forest: {rf_accuracy:.4f}")
print(f"SVM: {svm_accuracy:.4f}")


table = PrettyTable()
table.field_names = ["Model", "Precision", "Recall", "F1-score", "Accuracy"]

# Calculate metrics for each model
logreg_report = classification_report(y_test, logreg_predictions, output_dict=True)
knn_report = classification_report(y_test, knn_predictions, output_dict=True)
dt_report = classification_report(y_test, dt_predictions, output_dict=True)
rf_report = classification_report(y_test, rf_predictions, output_dict=True)
svm_report = classification_report(y_test, svm_predictions, output_dict=True)

# Add rows to the table
table.add_row(["Logistic Regression", logreg_report['weighted avg']['precision'], logreg_report['weighted avg']['recall'],
               logreg_report['weighted avg']['f1-score'], logreg_accuracy])

table.add_row(["kNN", knn_report['weighted avg']['precision'], knn_report['weighted avg']['recall'],
               knn_report['weighted avg']['f1-score'], knn_accuracy])

table.add_row(["Decision Tree", dt_report['weighted avg']['precision'], dt_report['weighted avg']['recall'],
               dt_report['weighted avg']['f1-score'], dt_accuracy])

table.add_row(["Random Forest", rf_report['weighted avg']['precision'], rf_report['weighted avg']['recall'],
               rf_report['weighted avg']['f1-score'], rf_accuracy])

table.add_row(["SVM", svm_report['weighted avg']['precision'], svm_report['weighted avg']['recall'],
               svm_report['weighted avg']['f1-score'], svm_accuracy])

print(table)
