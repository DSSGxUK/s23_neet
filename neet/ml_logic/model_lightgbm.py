import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit, GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, accuracy_score, fbeta_score, balanced_accuracy_score, classification_report, precision_recall_curve
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
import seaborn as sns
import shap
from scipy.special import softmax

class LightGBM:
    """
    Custom class for training, testing, and hyperparameter tuning using LightGBM.
    """

    def __init__(self,threshold=0.45):
        """
        Initialize the LightGBM class with default values.
        """
        self.num_folds = 5
        self.seed = 42
        self.scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f2_score': make_scorer(fbeta_score, beta=2),
            'balanced_accuracy': make_scorer(balanced_accuracy_score)
        }
        self.model = None
        self.model_name = "LightGBM"
        self.threshold = threshold

        
    def train_evaluate_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series) -> (lgb.LGBMClassifier, dict, dict):
        """
        Train and evaluate the LightGBM model.

        Parameters:
            X_train (pd.DataFrame): The input feature DataFrame for training.
            y_train (pd.Series): The target variable Series for training.

        Returns:
            best_model (lgb.LGBMClassifier): The trained LightGBM model.
            mean_score (dict): Mean scores (accuracy, F2 score, balanced accuracy) from cross-validation.
            std_scores (dict): Standard deviation of scores from cross-validation.
        """
        best_params = self.hyperparameter_search_lgbm(X_train, y_train)  # Get best hyperparameters

        train_data = lgb.Dataset(X_train, label=y_train)

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 67,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'is_unbalance': True,  # Automatically calculates scale_pos_weight
             **best_params 
        }

        model = lgb.LGBMClassifier(**params)

        kfold = StratifiedShuffleSplit(n_splits=self.num_folds, test_size=0.2, random_state=self.seed)
        cv_results = cross_validate(model, X_train, y_train, cv=kfold, scoring=self.scoring, return_train_score=True)

        mean_score = {
            'accuracy': cv_results["test_accuracy"].mean(),
            'f2_score': cv_results["test_f2_score"].mean(),
            'balanced_accuracy': cv_results["test_balanced_accuracy"].mean()
        }
        std_scores = {
            'accuracy': cv_results["test_accuracy"].std(),
            'f2_score': cv_results["test_f2_score"].std(),
            'balanced_accuracy': cv_results["test_balanced_accuracy"].std()
        }

        print(f"{self.model_name} - Mean scores:")
        for metric, score in mean_score.items():
            print(f"{metric}: {score:.2f}")

        self.model = model

        return model, mean_score, std_scores
    
    def test_model_lightgbm(self, X_test: pd.DataFrame, y_test: pd.Series, model: lgb.LGBMClassifier) -> dict:
        """
        Test the LightGBM model and evaluate its performance on the test data.

        Parameters:
            X_test (pd.DataFrame): The input feature DataFrame for testing.
            y_test (pd.Series): The target variable Series for testing.
            model (lgb.LGBMClassifier): The trained LightGBM model.

        Returns:
            scores (dict): Accuracy, F2 score, and balanced accuracy on the test data.
        """
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred >= self.threshold).astype(int)

        accuracy = accuracy_score(y_test, y_pred_binary)
        f2_score_val = fbeta_score(y_test, y_pred_binary, beta=2)
        balanced_acc_score = balanced_accuracy_score(y_test, y_pred_binary)

        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.show()

        class_report = classification_report(y_test, y_pred_binary)
        print("Classification Report:\n", class_report)

        scores = {
            'accuracy': accuracy,
            'f2_score': f2_score_val,
            'balanced_accuracy': balanced_acc_score

        }
        
        # Plot the confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred_binary)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix on Test Set')
        plt.show()
        
        return scores

    def hyperparameter_search_lgbm(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """ 
        Perform hyperparameter search for the LightGBM model using GridSearchCV.

        Parameters:
            X_train (pd.DataFrame): The input feature DataFrame for training.
            y_train (pd.Series): The target variable Series for training.

        Returns:
            best_params (dict): Optimal hyperparameters found through GridSearchCV.
        """
        model = lgb.LGBMClassifier()

        lgbm_params = {
            'boosting_type': ['gbdt', 'dart'],
            'num_leaves': [31, 50, 100],
            'learning_rate': [0.001,0.005,0.01, 0.05, 0.1]
        }

        grid_search = GridSearchCV(estimator=model, param_grid=lgbm_params, scoring=self.scoring,
                                   cv=self.num_folds, refit='accuracy')
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print("Best Hyperparameters:", best_params)

        return best_params


    
    def search(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """
        Perform hyperparameter search for the LightGBM model.

        Parameters:
            X_train (pd.DataFrame): The input feature DataFrame containing the training features.
            y_train (pd.Series): The input target variable Series containing the training labels.

        Returns:
            best_params (dict): Dictionary containing the best hyperparameters found during tuning.
        """
        best_params = self.hyperparameter_search_lgbm(X_train, y_train)
        return best_params
    


    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMClassifier:
        """
        Fit the LightGBM model to the training data using cross-validation.
    
        Parameters:
            X_train (pd.DataFrame): The input feature DataFrame containing the training features.
            y_train (pd.Series): The input target variable Series containing the training labels.
    
        Returns:
            model (lgb.LGBMClassifier): The trained LightGBM model.
        """
        sss = StratifiedShuffleSplit(n_splits=self.num_folds, test_size=0.2, random_state=self.seed)
        scores = []

        for train_idx, val_idx in sss.split(X_train, y_train):
            X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val_fold, y_val_fold = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
            train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
            val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
    
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9
            }
    
            model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[val_data])
    
            best_iteration = model.best_iteration
            y_pred = model.predict(X_val_fold, num_iteration=best_iteration)
            y_pred_binary = np.round(y_pred)
            accuracy = accuracy_score(y_val_fold, y_pred_binary)
            scores.append(accuracy)
    
        mean_accuracy = np.mean(scores)
        std_accuracy = np.std(scores)
    
        print(f"{self.model_name} - Mean Accuracy: {mean_accuracy:.2f} (std: {std_accuracy:.2f})")
    
        self.model = model
    
        return model


    def analyze_individual_rows(self, trained_model, X_train, X_test, y_test, X_test_stud_ids): 
        """
        Analyze individual row performance and important features using SHAP.

        Parameters:
            X (pd.DataFrame): The input feature DataFrame for testing.
            row_indices (list): List of row indices to analyze.
        """
        
        # Initialize the SHAP explainer
        explainer = shap.TreeExplainer(trained_model, feature_perturbation="tree_path_dependent")
        
        # Calculate SHAP values for all predictions
        shap_values = explainer.shap_values(X)
 
        return shap_values
