import map_gen
import json


rooms = json.load(open("room_geometry.json", "r"))
engine = map_gen.Engine(rooms, map_size=(72, 72), batch_size=4)

