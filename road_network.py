# This describes the road setup we use
class RoadNetwork:
    def __init__(self):
        self.road_name = "I-10 El Paso"
        self.directions = ["East", "West"]

    def describe(self):
        print("Road network")
        print("------------")
        print(f"{self.road_name} | Directions: {', '.join(self.directions)}")