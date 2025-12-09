from ev_charging_mlops_platform.data_ingest import run_ingest
from ev_charging_mlops_platform.feature_engineering import get_train_test



def test_feature_pipeline_runs():
    # Should not raise
    run_ingest()
    X_train, X_test, y_train, y_test = get_train_test()

    assert len(X_train) > 0
    assert X_train.shape[1] > 0
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
