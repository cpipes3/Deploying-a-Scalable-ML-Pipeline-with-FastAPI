import pytest
import pandas as pd
from ml.data import process_data
from ml.model import train_model
from ml.model import compute_model_metrics


def test_lables():
    """
    Test that salary labels are being applied
    """
    # Create test dataframe
    test_df = pd.DataFrame({
        'age' : [43, 22, 65, 32, 45],
        'race' : ['white', 'asian-pac-island', 'other', 'black', "white"],
        'sex' : ['female', 'female', 'male', 'male', 'male'],
        'education_num' : [16, 14, 16, 18, 14],
        'income' : ['>50K', '<=50K','<=50K', '>50K', '<=50K']
    })
    X, y, encoder, lb = process_data(test_df, ['race', 'sex'], 'income', training=True)

    assert X.shape[0] == len(test_df)
    assert y.shape[0] == len(X)
    assert set(y) <= {0,1}


def test_model_returns():
    """
    Test that a model object is returned
    """
    # Create test dataframe
    test_df = pd.DataFrame({
        'age' : [43, 22, 65, 32, 45],
        'race' : ['white', 'asian-pac-island', 'other', 'black', "white"],
        'sex' : ['female', 'female', 'male', 'male', 'male'],
        'education_num' : [16, 14, 16, 18, 14],
        'income' : ['>50K', '<=50K','<=50K', '>50K', '<=50K']
    })

    X_train, y_train, encoder, lb = process_data(test_df, ['race', 'sex'], 'income', training=True)

    model = train_model(X_train, y_train)

    assert model is not None




def test_compute_metrics():
    """
    Test that retuned values are as expected
    """
    # Set test labels and predicted labels to match
    test_labels = [0,1,1,0,1]
    test_preds = [0,1,1,0,1]

    precision, recall, f1 = compute_model_metrics(test_labels, test_preds)

    assert precision > 0.9
    assert recall > 0.9
    assert f1 > 0.9


