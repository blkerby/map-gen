"""Explicit v18 controller layout used only by checkpoint migration tools."""

import torch

from model import BalanceModel, activation_dtype, zero_init_output_layer
from model_loading import balance_model_kwargs


class BalanceModelV18(BalanceModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        failure_count = (
            self.left_variant_count + self.right_variant_count
            + self.up_variant_count + self.down_variant_count
        )
        for network, removed in (
            (self.door_net, failure_count),
            (self.area_net, self.num_room_connection_variants),
        ):
            output = torch.nn.Linear(
                network[-1].in_features, network[-1].out_features - removed
            )
            zero_init_output_layer(output)
            network[-1] = output

    def forward(self, generation_variable_floats):
        inputs = generation_variable_floats.to(activation_dtype(
            generation_variable_floats.device, next(self.parameters()).dtype,
        ))
        door = self.door_net(inputs).float()
        toilet = self.toilet_net(inputs).float()
        area = self.area_net(inputs).float()
        failure_count = (
            self.left_variant_count + self.right_variant_count
            + self.up_variant_count + self.down_variant_count
        )
        mean = (
            toilet[:, :self.num_rooms] * self.toilet_compatibility
        ).sum(-1) / self.toilet_compatibility.sum().clamp_min(1)
        # Adapt old raw failure output to the current success-relative price convention.
        toilet = torch.cat((toilet[:, :self.num_rooms], (toilet[:, -1] - mean).unsqueeze(1)), dim=1)
        return self.decode_prices(
            torch.cat((door, door.new_zeros((len(door), failure_count))), dim=1),
            toilet,
            torch.cat((area, area.new_zeros((len(area), self.num_room_connection_variants))), dim=1),
        )


def create_balance_model_v18(config, rooms, engine, device):
    return BalanceModelV18(**balance_model_kwargs(config, rooms, engine)).to(device)
