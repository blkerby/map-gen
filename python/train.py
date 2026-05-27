import map_gen
import json
import time
import numpy as np
import torch

from model import CausalTransformerModel
from generate import generate, GenerationConfig

# rooms_str = open("room_definitions/crateria.json", "r").read()
rooms_str = open("room_definitions/zebes.json", "r").read()
rooms = json.loads(rooms_str)
num_environments = 4
num_rounds = 1
max_candidates = 32
map_size = (72, 72)
temperature = 1.0
device = torch.device("cpu")

engine = map_gen.Engine(rooms_str)
env = engine.create_environment_group(map_size, num_environments, seed=6)

num_doors, num_connects = engine.get_output_sizes()
num_outputs = num_doors + num_connects

embedding_width = 256
key_width = 64
value_width = 64
attn_heads = 16
head_groups = 4
hidden_width = 512
num_layers = 4

main_model = CausalTransformerModel(
    num_rooms=len(rooms),
    map_x=map_size[0],
    map_y=map_size[1],
    num_outputs=num_outputs,
    embedding_width=embedding_width,
    key_width=key_width,
    value_width=value_width,
    attn_heads=attn_heads,
    head_groups=head_groups,
    hidden_width=hidden_width,
    num_layers=num_layers,
)

config = GenerationConfig(
    episode_length=len(rooms),
    max_candidates=max_candidates,
    temperature=torch.full([num_environments], temperature, dtype=torch.float32),
)

start = time.perf_counter()
for _ in range(num_rounds):
    actions, outcomes = generate(env, main_model, config, device)
    door_invalid, connection_invalid = outcomes
    door_invalid = np.count_nonzero(door_invalid, axis=1)
    connection_invalid = np.count_nonzero(connection_invalid, axis=1)
    print(f"Door invalid: {door_invalid}, Connection invalid: {connection_invalid}")

end = time.perf_counter()
print(f"Elapsed time: {(end - start):.3f} seconds, {(end - start)/(num_rounds*num_environments):.5f} seconds per episode")
