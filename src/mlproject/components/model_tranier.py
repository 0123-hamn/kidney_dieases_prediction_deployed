import os
import sys
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object



@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")



class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:
            logging.info("Splitting training and testing data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

           
            models = {
                "LogisticRegression": LogisticRegression(
                    max_iter=1000,
                    C=0.5,
                    class_weight="balanced"
                ),

                "RandomForest": RandomForestClassifier(
                    n_estimators=80,
                    max_depth=5,
                    min_samples_leaf=5,
                    random_state=42,
                    class_weight="balanced"
                ),

                "DecisionTree": DecisionTreeClassifier(
                    max_depth=4,
                    min_samples_leaf=8,
                    random_state=42,
                    class_weight="balanced"
                ),

                "KNN": KNeighborsClassifier(),

                "BernoulliNB": BernoulliNB(),

                "SVC": SVC(
                    probability=True,
                    C=0.5,
                    class_weight="balanced"
                )
            }

          
            params = {
                "LogisticRegression": {
                    "C": [0.01, 0.1, 0.5, 1]
                },

                "RandomForest": {
                    "n_estimators": [50, 80],
                    "max_depth": [4, 5],
                    "min_samples_leaf": [3, 5]
                },

                "DecisionTree": {
                    "max_depth": [3, 4, 5],
                    "min_samples_leaf": [5, 8]
                },

                "KNN": {
                    "n_neighbors": [3, 5, 7]
                },

                "BernoulliNB": {
                    "alpha": [0.1, 0.5, 1.0]
                },

                "SVC": {
                    "C": [0.1, 0.5, 1],
                    "kernel": ["linear", "rbf"]
                }
            }

            cv = RepeatedStratifiedKFold(
                n_splits=5,
                n_repeats=3,
                random_state=42
            )

            model_scores = {}
            trained_models = {}

            logging.info("Starting ROC-AUC based evaluation")

            
            for model_name, model in models.items():
                logging.info(f"Training {model_name}")

                grid = GridSearchCV(
                    model,
                    params[model_name],
                    scoring="roc_auc",
                    cv=cv,
                    n_jobs=-1
                )

                grid.fit(X_train, y_train)

                best_model = grid.best_estimator_
                trained_models[model_name] = best_model

               
                if hasattr(best_model, "predict_proba"):
                    y_probs = best_model.predict_proba(X_test)[:, 1]
                else:
                    y_probs = best_model.decision_function(X_test)

                roc_score = roc_auc_score(y_test, y_probs)
                model_scores[model_name] = roc_score

                logging.info(f"{model_name} Test ROC-AUC: {roc_score:.4f}")

                logging.info(
                    "\n" + classification_report(
                        y_test,
                        best_model.predict(X_test)
                    )
                )

            
            best_model_name = max(model_scores, key=model_scores.get)
            best_model = trained_models[best_model_name]
            best_model_score = model_scores[best_model_name]

            logging.info(
                f"Best Medical Model: {best_model_name} | Test ROC-AUC: {best_model_score:.4f}"
            )

           
            cv_scores = cross_val_score(
                best_model,
                X_train,
                y_train,
                cv=5,
                scoring="roc_auc"
            )

            logging.info(f"Cross-Validation ROC-AUC: {cv_scores.mean():.4f}")

            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                exist_ok=True
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Model saved successfully")

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)