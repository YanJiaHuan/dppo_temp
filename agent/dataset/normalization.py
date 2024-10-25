import pickle
import torch
import numpy as np
from collections import OrderedDict

# Replace 'data.pkl' with the path to your .pkl file
pkl_file_path = './2024-10-23-10:33:08.pkl'  # Update this with your actual .pkl file path
output_pth_path = './normalization.pth'

# Step 1: Load data from the .pkl file
with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f)

# Depending on how your data is structured, adjust accordingly.
# Assuming data is a tuple or list: (observations_list, actions_list)
if isinstance(data, dict):
    observations_list = data['observations']
    actions_list = data['actions']
elif isinstance(data, (list, tuple)) and len(data) == 2:
    observations_list = data[0]
    actions_list = data[1]
else:
    raise ValueError('Unexpected data format in the .pkl file')

# Initialize lists to store robot_states and actions
robot_states = []
actions = []

# Step 2: Extract robot_state from each observation
for obs in observations_list:
    # 'obs' is a dictionary with keys 'color_image1', 'color_image2', 'robot_state'
    robot_state = obs['robot_state']
    robot_states.append(robot_state)

# Convert lists to numpy arrays
robot_states = np.array(robot_states, dtype=np.float32)
actions = np.array(actions_list, dtype=np.float32)

# Step 3: Compute min and max statistics
# For robot_states (observations)
obs_min = np.min(robot_states, axis=0)
obs_max = np.max(robot_states, axis=0)

# For actions
act_min = np.min(actions, axis=0)
act_max = np.max(actions, axis=0)

# Step 4: Save the statistics into 'normalization.pth'
stats = OrderedDict()
stats['stats.observations.max'] = torch.tensor(obs_max)
stats['stats.observations.min'] = torch.tensor(obs_min)
stats['stats.actions.max'] = torch.tensor(act_max)
stats['stats.actions.min'] = torch.tensor(act_min)

# Save the stats dictionary to a .pth file
torch.save(stats, output_pth_path)

print(f"Normalization statistics have been saved to '{output_pth_path}'.")
