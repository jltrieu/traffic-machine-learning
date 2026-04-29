#This program defines the road system itself, like how they influence eachother

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class RoadSegment:
    roadName: str
    roadKind: str  # "mainline" or "feeder"
    upstreamMainline: Optional[str] = None


class TrafficRoadNetwork:
    def __init__(self):
        self.roadSegments: Dict[str, RoadSegment] = {
            "i10West": RoadSegment("i10West", "mainline", None),
            "i10Central": RoadSegment("i10Central", "mainline", "i10West"),
            "i10East": RoadSegment("i10East", "mainline", "i10Central"),
            "mesa": RoadSegment("mesa", "feeder"),
            "sunland": RoadSegment("sunland", "feeder"),
            "us54": RoadSegment("us54", "feeder"),
            "paisano": RoadSegment("paisano", "feeder"),
            "loop375": RoadSegment("loop375", "feeder"),
        }

        # If A points to B, traffic in A can affect B
        self.networkEdges: Dict[str, List[str]] = {
            "mesa": ["i10West"],
            "sunland": ["i10West"],
            "i10West": ["i10Central"],
            "us54": ["i10Central"],
            "paisano": ["i10Central"],
            "i10Central": ["i10East"],
            "loop375": ["i10East"],
            "i10East": [],
        }

        self.incomingEdgeMap = self._buildIncomingEdgeMap()

    def _buildIncomingEdgeMap(self) -> Dict[str, List[str]]:
        incomingEdgeMap = {roadName: [] for roadName in self.roadSegments.keys()}

        for sourceRoad, destinationRoads in self.networkEdges.items():
            for destinationRoad in destinationRoads:
                incomingEdgeMap[destinationRoad].append(sourceRoad)

        return incomingEdgeMap

    def getAllRoads(self) -> List[str]:
        return list(self.roadSegments.keys())

    def getIncomingRoads(self, targetRoad: str) -> List[str]:
        return list(self.incomingEdgeMap.get(targetRoad, []))

    def getOutgoingRoads(self, sourceRoad: str) -> List[str]:
        return list(self.networkEdges.get(sourceRoad, []))

    def getUpstreamMainline(self, targetRoad: str) -> Optional[str]:
        return self.roadSegments[targetRoad].upstreamMainline

    def getFeederRoads(self, targetRoad: str) -> List[str]:
        upstreamMainline = self.getUpstreamMainline(targetRoad)
        feederRoads = []

        for roadName in self.getIncomingRoads(targetRoad):
            if roadName != upstreamMainline:
                feederRoads.append(roadName)

        return feederRoads

    def printNetwork(self) -> None:
        print("Traffic road network:")
        for sourceRoad in self.getAllRoads():
            outgoingRoads = self.getOutgoingRoads(sourceRoad)
            if not outgoingRoads:
                print(f"  {sourceRoad} -> (none)")
            else:
                for destinationRoad in outgoingRoads:
                    print(f"  {sourceRoad} -> {destinationRoad}")


if __name__ == "__main__":
    trafficRoadNetwork = TrafficRoadNetwork()
    trafficRoadNetwork.printNetwork()
