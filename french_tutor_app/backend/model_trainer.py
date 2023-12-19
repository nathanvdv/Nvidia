import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from ast import literal_eval
import joblib


class ModelTrainer:
    def __init__(self, training_data_path, test_data_path):
        self.training_data_path = training_data_path
        self.test_data_path = test_data_path

    def load_data(self, path):
        data = pd.read_csv(path)
        data['embeddings'] = data['embeddings'].apply(literal_eval)

        # Creating a DataFrame from embeddings
        embeddings_df = pd.DataFrame(data['embeddings'].tolist(), index=data.index)
        embeddings_df.columns = [f'emb_{i}' for i in range(embeddings_df.shape[1])]

        # Concatenate the original data with the new embeddings DataFrame
        data = pd.concat([data, embeddings_df], axis=1).drop(['embeddings'], axis=1)
        return data

    def train_and_evaluate(self):
        # Load preprocessed training data
        training_data = self.load_data(self.training_data_path)
        test_data = self.load_data(self.test_data_path)
        # Dropping unnecessary columns
        training_data = training_data.drop(['id', 'sentence', 'cleaned_sentence', 'difficulty'], axis=1)
        test_data = test_data.drop(['id', 'sentence', 'cleaned_sentence'], axis=1)

        # Separate the features and target variable
        X_train = training_data.drop('difficulty_encoded', axis=1)
        y_train = training_data['difficulty_encoded']

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Initialize model and perform cross-validation
        svm_model = SVC(kernel='rbf')
        cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5)
        print(f"Cross-Validation Scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean()}, Standard Deviation: {cv_scores.std()}")

        # Hyperparameter tuning
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}
        grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, verbose=2, n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        print("Best parameters found: ", grid_search.best_params_)

        # Use the best estimator for predictions on test data
        best_svm = grid_search.best_estimator_
        X_test_scaled = scaler.transform(test_data)  # Standardizing test data
        model_filename = 'best_svm_model.joblib'
        joblib.dump(best_svm, model_filename)

        test_predictions = best_svm.predict(X_test_scaled)
        for idx, prediction in enumerate(test_predictions):
            print(f"Test Data ID {idx}: Predicted Difficulty: {prediction}")

        # Mapping predictions to CEFR levels and saving
        cefr_mapping = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}
        test_data_with_id = pd.read_csv(self.test_data_path)
        test_data_with_id['predicted_difficulty'] = test_predictions
        test_data_with_id['predicted_difficulty'] = test_data_with_id['predicted_difficulty'].map(cefr_mapping)
        test_data_with_id[['id', 'predicted_difficulty']].to_csv('french_tutor_app/backend/data/Nvidia_submission.csv', index=False)
        print("Predictions saved to Nvidia_submission.csv")

def main():
    trainer = ModelTrainer(
        training_data_path='data/Cleaned_Enhanced_Encoded_Training.csv',
        test_data_path='data/Cleaned_Enhanced_test.csv'
    )
    trainer.train_and_evaluate()

if __name__ == "__main__":
    main()

