import map_gen
import json

rooms = json.load(open("room_geometry.json", "r"))
# for x in rooms:
    # print(x['map'])
rooms = [rooms[0]]
print(rooms)
engine = map_gen.Engine(rooms)