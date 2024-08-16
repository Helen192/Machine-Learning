import pytest
import sys
import os
from pathlib import Path

# Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset
from prediction_model.predict import generate_predictions

# output from predict script is not null
# output from predict script is str data type
# the output is Y for an example data

# Fixture --> this function will run before execution of this each test function
# by mark the single_prediction function with pytest.fixture decorator, we can use this function in other test functions

@pytest.fixture
def single_prediction():
    test_dataset = load_dataset(config.TEST_FILE)
    single_row = test_dataset[:1]
    result = generate_predictions(single_row)
    return result

# output is not None
def test_single_pred_not_none(single_prediction):
    assert single_prediction is not None

# data type is string
def teest_single_pred_str(single_prediction):
    assert isinstance(single_prediction.get('Predictions')[0], str)

def test_single_pred_validate(single_prediction):
    assert single_prediction.get('Predictions')[0] == 'Y'