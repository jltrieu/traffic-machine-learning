#This should be our main ML program, it trains, test and predicts
#Should return in console, maybe?
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from trafficRoadNetwork import TrafficRoadNetwork
from TrafficDataProcessor import TrafficDataProcessor


class TrafficModeling:
    def __init__(self):
        self.trafficRoadNetwork = TrafficRoadNetwork()
        self.trafficDataProcessor = TrafficDataProcessor(self.trafficRoadNetwork)
        self.model = LogisticRegression(max_iter=1000)

    def chronologicalSplit(
        self,
        trainingTable: pd.DataFrame,
        trainSize: int = 250,
        testSize: int = 50,
    ):
        orderedTable = trainingTable.sort_values("timestamp").reset_index(drop=True)

        if len(orderedTable) < (trainSize + testSize):
            raise ValueError(
                f"Need at least {trainSize + testSize} rows, got {len(orderedTable)}"
            )

        trainTable = orderedTable.iloc[:trainSize].copy()
        testTable = orderedTable.iloc[trainSize : trainSize + testSize].copy()

        return trainTable, testTable

    def trainAndEvaluate(
        self,
        csvPath: str,
        targetRoad: str = "i10Central",
        horizonHours: int = 1,
        trainSize: int = 250,
        testSize: int = 50,
    ):
        rawDataFrame = self.trafficDataProcessor.loadCsv(csvPath)
        trainingTable = self.trafficDataProcessor.buildTrainingTable(
            rawDataFrame=rawDataFrame,
            targetRoad=targetRoad,
            horizonHours=horizonHours,
        )

        if trainingTable.empty:
            print("No usable training data was produced.")
            return

        featureColumns = self.trafficDataProcessor.getFeatureColumns(targetRoad)
        trainTable, testTable = self.chronologicalSplit(
            trainingTable, trainSize, testSize
        )

        xTrain = trainTable[featureColumns].fillna(0.0)
        yTrain = trainTable["targetIsCongested"].astype(int)

        xTest = testTable[featureColumns].fillna(0.0)
        yTest = testTable["targetIsCongested"].astype(int)

        self.model.fit(xTrain, yTrain)

        predictions = self.model.predict(xTest)
        probabilities = self.model.predict_proba(xTest)[:, 1]

        print(f"Target road: {targetRoad}")
        print(f"Horizon hours: {horizonHours}")
        print(f"Training rows: {len(trainTable)}")
        print(f"Test rows: {len(testTable)}")
        print(f"Accuracy: {accuracy_score(yTest, predictions):.4f}")
        print()
        print(classification_report(yTest, predictions, zero_division=0))
        print("Confusion matrix:")
        print(confusion_matrix(yTest, predictions))

        comparisonTable = testTable[["timestamp"]].copy()
        comparisonTable["actual"] = yTest.values
        comparisonTable["predicted"] = predictions
        comparisonTable["probabilityCongested"] = probabilities

        print("\nFirst 20 predictions:")
        print(comparisonTable.head(20).to_string(index=False))

    def predictLatest(
        self,
        csvPath: str,
        targetRoad: str = "i10Central",
        horizonHours: int = 1,
        trainSize: int = 250,
        testSize: int = 50,
    ):
        rawDataFrame = self.trafficDataProcessor.loadCsv(csvPath)

        trainingTable = self.trafficDataProcessor.buildTrainingTable(
            rawDataFrame=rawDataFrame,
            targetRoad=targetRoad,
            horizonHours=horizonHours,
        )

        if len(trainingTable) < (trainSize + testSize):
            print("Not enough data to train before predicting.")
            return

        featureColumns = self.trafficDataProcessor.getFeatureColumns(targetRoad)
        trainTable, _ = self.chronologicalSplit(trainingTable, trainSize, testSize)

        xTrain = trainTable[featureColumns].fillna(0.0)
        yTrain = trainTable["targetIsCongested"].astype(int)

        self.model.fit(xTrain, yTrain)

        latestFeatureRow = self.trafficDataProcessor.buildLatestFeatureRow(
            rawDataFrame=rawDataFrame,
            targetRoad=targetRoad,
        )

        latestFeatureRow = latestFeatureRow[featureColumns].fillna(0.0)

        predictedClass = int(self.model.predict(latestFeatureRow)[0])
        predictedProbability = float(self.model.predict_proba(latestFeatureRow)[0, 1])

        print("Latest prediction:")
        print(f"  targetRoad: {targetRoad}")
        print(f"  predictedCongested: {predictedClass}")
        print(f"  predictedProbability: {predictedProbability:.4f}")
        print("  featureValues:")
        for key, value in latestFeatureRow.iloc[0].to_dict().items():
            print(f"    {key}: {value}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python trafficModeling.py train <csvPath>")
        print("  python trafficModeling.py predict <csvPath>")
        sys.exit(1)

    command = sys.argv[1].strip().lower()
    csvPath = sys.argv[2].strip()

    trafficModeling = TrafficModeling()

    if command == "train":
        trafficModeling.trainAndEvaluate(csvPath=csvPath, targetRoad="i10Central")
    elif command == "predict":
        trafficModeling.predictLatest(csvPath=csvPath, targetRoad="i10Central")
    else:
        print(f"Unknown command: {command}")
        print("Use 'train' or 'predict'.")

# python trafficModeling.py train TrafficData.csv
# python trafficModeling.py predict TrafficData.csv