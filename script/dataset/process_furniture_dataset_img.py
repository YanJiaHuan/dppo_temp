import os
import pickle
import numpy as np
from tqdm import tqdm
import argparse

def make_dataset(load_dir, save_dir, save_name_prefix, val_split=0.0, normalize=False):
    # Get a list of all .pkl files in the directory
    pkl_files = [os.path.join(load_dir, f) for f in os.listdir(load_dir) if f.endswith('.pkl')]
    pkl_files.sort()  # Sort the files if needed

    # Initialize lists to hold data
    all_states = []
    all_actions = []
    all_images = []
    traj_lengths = []

    # For normalization (if needed)
    obs_min = None
    obs_max = None
    action_min = None
    action_max = None

    # Process each .pkl file
    for pkl_file in tqdm(pkl_files, desc='Processing .pkl files'):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        # Extract observations and actions
        observations = data['observations']
        actions = data['actions']

        # Extract robot states and images from observations
        states = []
        images = []
        for obs in observations:
            # Extract robot state
            state = obs['robot_state']
            states.append(state)

            # Extract images and combine them
            img1 = obs['color_image1']  # Shape: (3, H, W)
            img2 = obs['color_image2']  # Shape: (3, H, W)

            # Stack images along a new dimension (e.g., channel or time)
            combined_images = np.stack([img1, img2], axis=0)  # Shape: (2, 3, H, W)
            images.append(combined_images)

        # Convert lists to numpy arrays
        states = np.array(states)  # Shape: (T, state_dim)
        actions = np.array(actions)  # Shape: (T, action_dim)
        images = np.array(images)  # Shape: (T, 2, 3, H, W)

        # Update normalization stats
        if normalize:
            if obs_min is None:
                obs_min = np.min(states, axis=0)
                obs_max = np.max(states, axis=0)
                action_min = np.min(actions, axis=0)
                action_max = np.max(actions, axis=0)
            else:
                obs_min = np.minimum(obs_min, np.min(states, axis=0))
                obs_max = np.maximum(obs_max, np.max(states, axis=0))
                action_min = np.minimum(action_min, np.min(actions, axis=0))
                action_max = np.maximum(action_max, np.max(actions, axis=0))

        # Append to the lists
        all_states.append(states)
        all_actions.append(actions)
        all_images.append(images)
        traj_lengths.append(len(states))

    # Concatenate all data
    all_states = np.concatenate(all_states, axis=0)  # Shape: (total_transitions, state_dim)
    all_actions = np.concatenate(all_actions, axis=0)  # Shape: (total_transitions, action_dim)
    all_images = np.concatenate(all_images, axis=0)  # Shape: (total_transitions, 2, 3, H, W)
    traj_lengths = np.array(traj_lengths)

    # Normalize if needed
    if normalize:
        all_states = 2 * (all_states - obs_min) / (obs_max - obs_min + 1e-6) - 1
        all_actions = 2 * (all_actions - action_min) / (action_max - action_min + 1e-6) - 1
        # Images are typically already normalized; adjust if necessary.

    # Split into train and validation sets if val_split > 0
    num_traj = len(traj_lengths)
    num_train = int(num_traj * (1 - val_split))
    indices = np.arange(num_traj)
    np.random.shuffle(indices)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    # Initialize output dictionaries
    out_train = {'states': [], 'actions': [], 'images': [], 'traj_lengths': []}
    out_val = {'states': [], 'actions': [], 'images': [], 'traj_lengths': []}

    # Split data
    start_idx = 0
    for i in range(num_traj):
        end_idx = start_idx + traj_lengths[i]
        if i in train_indices:
            out = out_train
        else:
            out = out_val
        out['states'].append(all_states[start_idx:end_idx])
        out['actions'].append(all_actions[start_idx:end_idx])
        out['images'].append(all_images[start_idx:end_idx])
        out['traj_lengths'].append(traj_lengths[i])
        start_idx = end_idx

    # Concatenate lists
    for key in ['states', 'actions', 'images']:
        out_train[key] = np.concatenate(out_train[key], axis=0)
        if val_split > 0 and len(out_val[key]) > 0:
            out_val[key] = np.concatenate(out_val[key], axis=0)

    # Save datasets
    train_save_path = os.path.join(save_dir, save_name_prefix + 'train.npz')
    np.savez_compressed(
        train_save_path,
        states=out_train['states'],
        actions=out_train['actions'],
        images=out_train['images'],
        traj_lengths=np.array(out_train['traj_lengths']),
    )

    if val_split > 0 and len(out_val['states']) > 0:
        val_save_path = os.path.join(save_dir, save_name_prefix + 'val.npz')
        np.savez_compressed(
            val_save_path,
            states=out_val['states'],
            actions=out_val['actions'],
            images=out_val['images'],
            traj_lengths=np.array(out_val['traj_lengths']),
        )

    # Save normalization stats if needed
    if normalize:
        normalization_save_path = os.path.join(save_dir, save_name_prefix + 'normalization.pth')
        np.savez_compressed(
            normalization_save_path,
            obs_min=obs_min,
            obs_max=obs_max,
            action_min=action_min,
            action_max=action_max,
        )

    # Print summary
    print(f'Total trajectories: {num_traj}')
    print(f'Total transitions: {len(all_states)}')
    print(f'Train trajectories: {len(out_train["traj_lengths"])}')
    print(f'Validation trajectories: {len(out_val["traj_lengths"])}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', type=str, required=True, help='Directory containing .pkl files')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the .npz files')
    parser.add_argument('--save_name_prefix', type=str, default='', help='Prefix for saved .npz files')
    parser.add_argument('--val_split', type=float, default=0.0, help='Fraction of data to use for validation')
    parser.add_argument('--normalize', action='store_true', help='Whether to normalize data')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    make_dataset(
        load_dir=args.load_dir,
        save_dir=args.save_dir,
        save_name_prefix=args.save_name_prefix,
        val_split=args.val_split,
        normalize=args.normalize,
    )
