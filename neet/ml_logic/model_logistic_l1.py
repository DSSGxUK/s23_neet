"""
Custom Logistic Regression with L1 Penalty  Model for Predicting Young People at risk of becoming NEET
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, fbeta_score, classification_report, precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix





class LogisticRegressionL1():
    """
    Custom implementation of Logistic Regression with L1 penalty.

    This class encapsulates the training, hyperparameter tuning, testing, and prediction steps
    for a Logistic Regression model with L1 penalty.

    Parameters:
    threshold (float, optional): The threshold value to adjust the decision threshold for classifying samples
                                 as positive or negative. Defaults to 0.4.
    """

    def __init__(self, threshold=0.6):
        self.threshold = threshold
        self.model = None

    def train_evaluate_logistic(self, X_train, y_train, threshold=None):
        """
        Train and evaluate a Logistic Regression model using 5-fold cross-validation.

        Parameters:
        X_train (DataFrame): The input pandas DataFrame containing the training features.
        y_train (Series): The input pandas Series containing the training target variable (labels).
        threshold (float, optional): The threshold value to adjust the decision threshold for classifying samples
                                     as positive or negative. Defaults to None.

        Returns:
        avg_balanced_accuracy (float): The average balanced accuracy score computed through 5-fold cross-validation.
        avg_f2_score (float): The average F2 score computed through 5-fold cross-validation.
        """
        if threshold is None:
            threshold = self.threshold
        
        # Get the best hyperparameters from hyperparameter tuning
        best_params = self.hyperparameter_tuning_logistic(X_train, y_train)            

        # Create an instance of StratifiedShuffleSplit for 5-fold cross-validation
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)

        # Initialize lists to store evaluation metrics
        balanced_accuracies = []
        f2_scores = []

        # Loop through the cross-validation splits on the train set
        for train_index, val_index in sss.split(X_train, y_train):
            # Split the data into training and validation sets for this split
            X_train_cv, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_cv, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

            # Create the Logistic Regression model with L1 penalty
            model = LogisticRegression(penalty='l1', solver='liblinear')

            model.fit(X_train_cv, y_train_cv)

            # Get the probability estimates for the positive class (class 1)
            y_probs = model.predict_proba(X_val)[:, 1]

            # Adjust the decision threshold to classify samples as positive or negative
            y_pred_binary = np.where(y_probs >= threshold, 1, 0)

            # Compute balanced accuracy and F2 score for this split
            balanced_accuracy = balanced_accuracy_score(y_val, y_pred_binary)
            f2_score = fbeta_score(y_val, y_pred_binary, beta=2)

            # Append the scores to the lists
            balanced_accuracies.append(balanced_accuracy)
            f2_scores.append(f2_score)

            # Print the classification report for this split
            print(f"Classification Report for Split {len(balanced_accuracies)}")
            print(classification_report(y_val, y_pred_binary))

        # Compute the average scores over all splits
        avg_balanced_accuracy = np.mean(balanced_accuracies)
        avg_f2_score = np.mean(f2_scores)

        self.model = model  # Save the trained model
        return avg_balanced_accuracy, avg_f2_score
    
    
    def hyperparameter_tuning_logistic(self, X_train, y_train):
        """
        Perform hyperparameter tuning for the Logistic Regression model.

        Parameters:
        X_train (DataFrame): The input pandas DataFrame containing the training features.
        y_train (Series): The input pandas Series containing the training target variable (labels).


        Returns:
        best_params (dict): A dictionary containing the best hyperparameters found during the hyperparameter tuning process.
        """
        # Specify the hyperparameters to tune and their possible values
        param_grid = {
            'C': [0.001, 0.01, 0.1, 0.5, 1, 2,3,4, 10, 100],
            'penalty': ['l1'],
            'solver': ['liblinear']
        }

        # Create the Logistic Regression model
        model = LogisticRegression(class_weight='balanced', max_iter=1000)

        # Initialize GridSearchCV with 5-fold cross-validation
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)

        # Perform hyperparameter tuning
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_params = grid_search.best_params_

        return best_params

    def test_logistic(self, X_test, y_test, threshold=None):
        """
        Test the Logistic Regression model on the test set and print the evaluation metrics.

        Parameters:
        X_test (DataFrame): The input pandas DataFrame containing the test features.
        y_test (Series): The input pandas Series containing the test target variable (labels).
        threshold (float, optional): The threshold value to adjust the decision threshold for classifying samples
                                     as positive or negative. Defaults to None.
        """
        if threshold is None:
            threshold = self.threshold
        
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Get the probability estimates for the positive class (class 1) on the test set
        y_probs = self.model.predict_proba(X_test)[:, 1]

        # Adjust the decision threshold to classify samples as positive or negative on the test set
        y_pred_binary = np.where(y_probs >= self.threshold, 1, 0)

        # Compute balanced accuracy and F2 score on the test set
        final_balanced_accuracy = balanced_accuracy_score(y_test, y_pred_binary)
        final_f2_score = fbeta_score(y_test, y_pred_binary, beta=2)

        # Print the final evaluation metrics on the test set
        print("Final Balanced Accuracy on Test Set:", final_balanced_accuracy)
        print("Final F2 Score on Test Set:", final_f2_score)

        # Print the final classification report on the test set
        print("Final Classification Report on Test Set")
        print(classification_report(y_test, y_pred_binary))

        # Plot the confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred_binary)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix on Test Set')
        plt.show()

        # Compute precision, recall, and thresholds on the test set
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_binary)
        print(f"precision- {precision[:5]}")
        print(f"recall- {recall[:5]}")
        print(f"thresholds- {thresholds[:5]}")

        # Plot the Precision-Recall curve on the test set
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve for Logistic Regression with L1 Penalty on Test Set')
        plt.grid(True)
        plt.show()
        return conf_matrix

    def fit(self, X_train, y_train, threshold=None):
        """
        Fit the Logistic Regression model to the training data.

        Parameters:
        X_train (DataFrame): The input pandas DataFrame containing the training features.
        y_train (Series): The input pandas Series containing the training target variable (labels).
        threshold (float, optional): The threshold value to adjust the decision threshold for classifying samples
                                     as positive or negative. Defaults to None.
        """
        self.train_evaluate_logistic(X_train, y_train, threshold)

    def search(self, X_train, y_train):
        """
        Perform hyperparameter search for the Logistic Regression model.

        Parameters:
        X_train (DataFrame): The input pandas DataFrame containing the training features.
        y_train (Series): The input pandas Series containing the training target variable (labels).

        Returns:
        best_params (dict): A dictionary containing the best hyperparameters found during the hyperparameter tuning process.
        """
        best_params = self.hyperparameter_tuning_logistic(X_train, y_train)
        return best_params

    

    def predict(self, X_test: pd.DataFrame, threshold=0.55):
        """
        Predict using the trained Logistic Regression model.

        Parameters:
            X_test (pd.DataFrame): The input feature DataFrame containing the test features.
            threshold (float): Decision threshold for binary classification. If None, the raw predictions are returned.

        Returns:
            y_pred (np.ndarray): Predicted labels for the test set.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please call the fit method first.")

        y_probs = self.model.predict_proba(X_test)[:, 1]

        if threshold is not None:
            y_pred_binary = np.where(y_probs >= threshold, 1, 0)
            return y_pred_binary
        else:
            return y_probs