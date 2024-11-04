"""
Normalization for Furniture-Bench environments.

TODO: use this normalizer for all benchmarks.

"""

import torch.nn as nn


class LinearNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.stats = nn.ParameterDict()
    def fit(self, data_dict):
        for key, tensor in data_dict.items():
            min_value = tensor.min(dim=0)[0]
            max_value = tensor.max(dim=0)[0]

            # Check if any column has only one value throughout
            diff = max_value - min_value
            constant_columns = diff == 0

            # Set a small range for constant columns to avoid division by zero
            min_value[constant_columns] -= 1
            max_value[constant_columns] += 1

            self.stats[key] = nn.ParameterDict(
                {
                    "min": nn.Parameter(min_value, requires_grad=False),
                    "max": nn.Parameter(max_value, requires_grad=False),
                },
            )
            print('self.stats[key]: ', self.stats[key])
        self._turn_off_gradients()

    def _normalize(self, x, key):
        stats = self.stats[key]
        # no matter what the stats shape is, the status should be same with x
        if stats["max"].shape[0] != x.shape[1]:
            stats["max"] = stats["max"][:x.shape[1]]
            stats["min"] = stats["min"][:x.shape[1]]
        x = (x - stats["min"]) / (stats["max"] - stats["min"])
        x = 2 * x - 1

        # print('key: ', key)
        # print('stats: ', stats)
        # print("status[max]'s shape is: ", stats["max"].shape)
        # print("status[min]'s shape is: ", stats["min"].shape)
        # print("x's shape is: ", x.shape)
        # x = x[:, :, :14] #58 modified 10.30
        # print("x's shape is: ", x.shape)
        # if stats["max"].shape[0] == 58:
        #     stats["max"] = stats["max"][:14]
        #     stats["min"] = stats["min"][:14]
        # x = (x - stats["min"]) / (stats["max"] - stats["min"])
        # x = 2 * x - 1
        return x

    def _denormalize(self, x, key):
        stats = self.stats[key]
        if stats["max"].shape[0] != x.shape[1]:
            stats["max"] = stats["max"][:x.shape[1]]
            stats["min"] = stats["min"][:x.shape[1]]
        x = (x + 1) / 2
        x = x * (stats["max"] - stats["min"]) + stats["min"]
        return x

    def forward(self, x, key, forward=True):
        if forward:
            return self._normalize(x, key)
        else:
            return self._denormalize(x, key)

    def _turn_off_gradients(self):
        for key in self.stats.keys():
            for stat in self.stats[key].keys():
                self.stats[key][stat].requires_grad = False

    def load_state_dict(self, state_dict):

        stats = nn.ParameterDict()
        for key, value in state_dict.items():
            if value[0].shape == 58:
                value = value[:14]
            if key.startswith("stats."):
                param_key = key[6:]
                keys = param_key.split(".")
                current_dict = stats
                for k in keys[:-1]:
                    if k not in current_dict:
                        current_dict[k] = nn.ParameterDict()
                    current_dict = current_dict[k]
                current_dict[keys[-1]] = nn.Parameter(value)

        self.stats = stats
        self._turn_off_gradients()

        return f"<Added keys {self.stats.keys()} to the normalizer.>"

    def keys(self):
        return self.stats.keys()
