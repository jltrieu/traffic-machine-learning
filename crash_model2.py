# Changes from v1:
#   - Replaced RandomForestClassifier with XGBClassifier
#   - Added SMOTE to oversample minority classes

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from road_network import RoadNetwork
from crash_data import (
    load_dataset,
    create_time_block_dataset,
    prepare_features_and_target,
    split_dataset,
)
# apply SMOTE to training data
def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print("\nSMOTE applied")
    print("-------------")
    from collections import Counter
    before = Counter(y_train)
    after  = Counter(y_resampled)
    names  = {0: "NONE", 1: "LOW", 2: "MEDIUM", 3: "HIGH"}
    for cls in [0, 1, 2, 3]:
        print(f"  {names[cls]:8s}: {before.get(cls, 0):5d} → {after.get(cls, 0):5d}")

    return X_resampled, y_resampled

# train XGBoost model
def train_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model

# predict on test set
def predict_test_set(model, X_test):
    predictions = model.predict(X_test)
    return predictions

# evaluate model performance
def evaluate_model(y_test, predictions):
    labels = [0, 1, 2, 3]
    label_names = ["NONE", "LOW", "MEDIUM", "HIGH"]

    accuracy = accuracy_score(y_test, predictions) * 100

    print("\nModel results")
    print("-------------")
    print(f"Accuracy: {accuracy:.2f}%")

    print("\nConfusion matrix")
    print("Rows = actual values")
    print("Columns = predicted values")
    print("Order: NONE, LOW, MEDIUM, HIGH")
    print(confusion_matrix(y_test, predictions, labels=labels))

    print("\nClassification report")
    print(
        classification_report(
            y_test,
            predictions,
            labels=labels,
            target_names=label_names,
            zero_division=0,
        )
    )

def main():
    csv_file = "i10_crash_risk.csv"
    roads = RoadNetwork()
    roads.describe()

    # load hourly dataset
    data = load_dataset(csv_file)

    # convert hourly data into time-block data
    block_data = create_time_block_dataset(data)

    # prepare X and y
    X, y = prepare_features_and_target(block_data)

    # split into train and test sets
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print("\nDataset loaded")
    print("--------------")
    print(f"Original hourly rows: {len(data)}")
    print(f"Time-block rows: {len(block_data)}")
    print(f"Training rows (before SMOTE): {len(X_train)}")
    print(f"Test rows: {len(X_test)}")

    # apply SMOTE to training data only
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)
    print(f"Training rows (after SMOTE):  {len(X_train_sm)}")

    # train XGBoost model on SMOTE data
    model = train_model(X_train_sm, y_train_sm)

    # predict on test set
    predictions = predict_test_set(model, X_test)

    # evaluate model performance
    evaluate_model(y_test, predictions)

if __name__ == "__main__":
    main()
