import os
import sys
import dill
import numpy as np
import pandas as pd
import pymysql

from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")


def read_sql_data():
    logging.info("Reading SQL database started")

    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )

        logging.info("Database connection established")

        df = pd.read_sql_query("SELECT * FROM kidney_disease", mydb)

        logging.info(f"Data fetched successfully. Shape: {df.shape}")

        mydb.close()

        return df

    except Exception as ex:
        raise CustomException(ex, sys)



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)



def evaluate_models(X_train, y_train, X_test, y_test, models, params):

    try:
        report = {}
        trained_models = {}

        
        cv = RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=2,
            random_state=42
        )

        for model_name, model in models.items():

            logging.info(f"Training {model_name}")

            param_grid = params[model_name]

            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1
            )

            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            trained_models[model_name] = best_model

            
            if hasattr(best_model, "predict_proba"):
                y_probs = best_model.predict_proba(X_test)[:, 1]
            else:
                y_probs = best_model.decision_function(X_test)

            roc_score = roc_auc_score(y_test, y_probs)
            report[model_name] = roc_score

            logging.info(f"{model_name} Test ROC-AUC: {roc_score:.4f}")

            
            cv_scores = cross_val_score(
                best_model,
                X_train,
                y_train,
                cv=5,
                scoring="roc_auc"
            )

            logging.info(f"{model_name} CV ROC-AUC: {cv_scores.mean():.4f}")

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)
