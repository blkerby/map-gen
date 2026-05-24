import map_gen
import json


rooms = json.load(open("room_geometry.json", "r"))
engine = map_gen.Engine(rooms)
