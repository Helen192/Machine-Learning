import pathlib
import os
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent  # /Users/username/PycharmProjects/Packaging-ML-Model/packaging-ml-model/prediction_model

DATAPATH = os.path.join(PACKAGE_ROOT, "datasets")  # /Users/username/PycharmProjects/Packaging-ML-Model/packaging-ml-model/prediction_model/datasets

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

MODEL_NAME = 'classification.pkl'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, "trained_models")  # /Users/username/PycharmProjects/Packaging-ML-Model/packaging-ml-model/prediction_model/trained_models

TARGET = "Loan_Status"

# Final features used in the mdoel
FEATURES = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 
            'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']

NUM_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

CAT_FEATURES = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']

# in our case it is same as Categorical features
FEATURES_TO_ENCODE = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']

FEATURE_TO_MODIFY = ['ApplicantIncome']
FEATURE_TO_ADD = 'CoapplicantIncome'

DROP_FEATURES = ['CoapplicantIncome']
LOG_FEATURES = ['ApplicantIncome', 'LoanAmount']