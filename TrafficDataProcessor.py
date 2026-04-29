#THis program handles the data, using a csv (hopefully)
#we may need to change some stuff here
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd

from trafficRoadNetwork import TrafficRoadNetwork


class TrafficDataProcessor:
    def __init__(self, trafficRoadNetwork: TrafficRoadNetwork):
        self.trafficRoadNetwork = trafficRoadNetwork
        self.congestionThreshold = 0.60

    def loadCsv(self, csvPath: str) -> pd.DataFrame:
        """
        Expected future raw CSV format:
            timestamp
            roadName
            congestionValue
            incidentCount
            rainFlag
            isHoliday

        We may change this later once we get the real data.
        """
        dataFrame = pd.read_csv(csvPath)
        return dataFrame

    def normalizeRawData(self, rawDataFrame: pd.DataFrame) -> pd.DataFrame:
        if rawDataFrame.empty:
            return rawDataFrame.copy()

        dataFrame = rawDataFrame.copy()

        requiredColumns = ["timestamp", "roadName"]
        for columnName in requiredColumns:
            if columnName not in dataFrame.columns:
                raise ValueError(f"Missing required column: {columnName}")

        dataFrame["timestamp"] = pd.to_datetime(dataFrame["timestamp"])

        if "congestionValue" not in dataFrame.columns:
            dataFrame["congestionValue"] = 0.0

        if "incidentCount" not in dataFrame.columns:
            dataFrame["incidentCount"] = 0

        if "rainFlag" not in dataFrame.columns:
            dataFrame["rainFlag"] = 0

        if "isHoliday" not in dataFrame.columns:
            dataFrame["isHoliday"] = 0

        dataFrame["hour"] = dataFrame["timestamp"].dt.hour
        dataFrame["dayOfWeek"] = dataFrame["timestamp"].dt.dayofweek
        dataFrame["isWeekend"] = (dataFrame["dayOfWeek"] >= 5).astype(int)
        dataFrame["isCongested"] = (
            dataFrame["congestionValue"] >= self.congestionThreshold
        ).astype(int)

        dataFrame["incidentCount"] = dataFrame["incidentCount"].fillna(0)
        dataFrame["rainFlag"] = dataFrame["rainFlag"].astype(int)
        dataFrame["isHoliday"] = dataFrame["isHoliday"].astype(int)

        dataFrame = dataFrame.sort_values(["roadName", "timestamp"]).reset_index(
            drop=True
        )
        return dataFrame

    def getFeatureColumns(self, targetRoad: str) -> List[str]:
        featureColumns = [
            "hour",
            "dayOfWeek",
            "isWeekend",
            "isHoliday",
            "rainFlag",
            "incidentCount",
            "ownCongestionNow",
            "ownCongestionMinus1",
            "ownCongestionMinus2",
            "ownCongestionMinus24",
            "upstreamCongestionMinus1",
        ]

        for feederRoad in self.trafficRoadNetwork.getFeederRoads(targetRoad):
            featureColumns.append(f"{feederRoad}PrevCongestion")

        return featureColumns

    def _lookupValue(
        self,
        lookupTable: pd.DataFrame,
        roadName: str,
        timestamp,
        columnName: str,
        defaultValue=np.nan,
    ):
        try:
            return lookupTable.loc[(roadName, timestamp), columnName]
        except KeyError:
            return defaultValue

    def buildTrainingTable(
        self,
        rawDataFrame: pd.DataFrame,
        targetRoad: str,
        horizonHours: int = 1,
    ) -> pd.DataFrame:
        dataFrame = self.normalizeRawData(rawDataFrame)

        if dataFrame.empty:
            outputColumns = (
                ["timestamp"]
                + self.getFeatureColumns(targetRoad)
                + ["targetIsCongested"]
            )
            return pd.DataFrame(columns=outputColumns)

        targetRows = dataFrame[dataFrame["roadName"] == targetRoad].copy()

        if targetRows.empty:
            outputColumns = (
                ["timestamp"]
                + self.getFeatureColumns(targetRoad)
                + ["targetIsCongested"]
            )
            return pd.DataFrame(columns=outputColumns)

        lookupTable = dataFrame.set_index(["roadName", "timestamp"]).sort_index()

        upstreamRoad = self.trafficRoadNetwork.getUpstreamMainline(targetRoad)
        feederRoads = self.trafficRoadNetwork.getFeederRoads(targetRoad)

        trainingRows = []

        for _, currentRow in targetRows.iterrows():
            currentTimestamp = currentRow["timestamp"]
            futureTimestamp = currentTimestamp + timedelta(hours=horizonHours)

            featureRow = {
                "timestamp": currentTimestamp,
                "hour": currentRow["hour"],
                "dayOfWeek": currentRow["dayOfWeek"],
                "isWeekend": currentRow["isWeekend"],
                "isHoliday": currentRow["isHoliday"],
                "rainFlag": currentRow["rainFlag"],
                "incidentCount": currentRow["incidentCount"],
                "ownCongestionNow": currentRow["congestionValue"],
                "ownCongestionMinus1": self._lookupValue(
                    lookupTable,
                    targetRoad,
                    currentTimestamp - timedelta(hours=1),
                    "congestionValue",
                ),
                "ownCongestionMinus2": self._lookupValue(
                    lookupTable,
                    targetRoad,
                    currentTimestamp - timedelta(hours=2),
                    "congestionValue",
                ),
                "ownCongestionMinus24": self._lookupValue(
                    lookupTable,
                    targetRoad,
                    currentTimestamp - timedelta(hours=24),
                    "congestionValue",
                ),
                "upstreamCongestionMinus1": (
                    0.0
                    if upstreamRoad is None
                    else self._lookupValue(
                        lookupTable,
                        upstreamRoad,
                        currentTimestamp - timedelta(hours=1),
                        "congestionValue",
                    )
                ),
                "targetIsCongested": self._lookupValue(
                    lookupTable, targetRoad, futureTimestamp, "isCongested"
                ),
            }

            for feederRoad in feederRoads:
                featureRow[f"{feederRoad}PrevCongestion"] = self._lookupValue(
                    lookupTable,
                    feederRoad,
                    currentTimestamp - timedelta(hours=1),
                    "congestionValue",
                )

            trainingRows.append(featureRow)

        trainingTable = pd.DataFrame(trainingRows)
        trainingTable = trainingTable.dropna(subset=["targetIsCongested"]).reset_index(
            drop=True
        )

        orderedColumns = (
            ["timestamp"] + self.getFeatureColumns(targetRoad) + ["targetIsCongested"]
        )
        return trainingTable[orderedColumns]

    def buildLatestFeatureRow(
        self,
        rawDataFrame: pd.DataFrame,
        targetRoad: str,
    ) -> pd.DataFrame:
        dataFrame = self.normalizeRawData(rawDataFrame)

        if dataFrame.empty:
            zeroRow = {
                columnName: 0.0 for columnName in self.getFeatureColumns(targetRoad)
            }
            return pd.DataFrame([zeroRow])

        targetRows = dataFrame[dataFrame["roadName"] == targetRoad].copy()
        if targetRows.empty:
            zeroRow = {
                columnName: 0.0 for columnName in self.getFeatureColumns(targetRoad)
            }
            return pd.DataFrame([zeroRow])

        lookupTable = dataFrame.set_index(["roadName", "timestamp"]).sort_index()
        latestTimestamp = targetRows["timestamp"].max()
        currentRow = targetRows[targetRows["timestamp"] == latestTimestamp].iloc[-1]

        upstreamRoad = self.trafficRoadNetwork.getUpstreamMainline(targetRoad)
        feederRoads = self.trafficRoadNetwork.getFeederRoads(targetRoad)

        featureRow = {
            "hour": currentRow["hour"],
            "dayOfWeek": currentRow["dayOfWeek"],
            "isWeekend": currentRow["isWeekend"],
            "isHoliday": currentRow["isHoliday"],
            "rainFlag": currentRow["rainFlag"],
            "incidentCount": currentRow["incidentCount"],
            "ownCongestionNow": currentRow["congestionValue"],
            "ownCongestionMinus1": self._lookupValue(
                lookupTable,
                targetRoad,
                latestTimestamp - timedelta(hours=1),
                "congestionValue",
                0.0,
            ),
            "ownCongestionMinus2": self._lookupValue(
                lookupTable,
                targetRoad,
                latestTimestamp - timedelta(hours=2),
                "congestionValue",
                0.0,
            ),
            "ownCongestionMinus24": self._lookupValue(
                lookupTable,
                targetRoad,
                latestTimestamp - timedelta(hours=24),
                "congestionValue",
                0.0,
            ),
            "upstreamCongestionMinus1": (
                0.0
                if upstreamRoad is None
                else self._lookupValue(
                    lookupTable,
                    upstreamRoad,
                    latestTimestamp - timedelta(hours=1),
                    "congestionValue",
                    0.0,
                )
            ),
        }

        for feederRoad in feederRoads:
            featureRow[f"{feederRoad}PrevCongestion"] = self._lookupValue(
                lookupTable,
                feederRoad,
                latestTimestamp - timedelta(hours=1),
                "congestionValue",
                0.0,
            )

        latestFeatureRow = pd.DataFrame([featureRow])
        return latestFeatureRow[self.getFeatureColumns(targetRoad)]
