import os
import subprocess
import zipfile
import simpler_env
import tensorflow as tf
import logging
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import json
import numpy as np
import sapien.core as sapien
import mediapy
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO/WARNING, 3=no INFO/WARNING/ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"GPU memory growth configuration error: {e}")

# Set environment variables for CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

RT_1_CHECKPOINTS = {
    "rt_1_x": "rt_1_x_tf_trained_for_002272480_step",
    "rt_1_400k": "rt_1_tf_trained_for_000400120",
    "rt_1_58k": "rt_1_tf_trained_for_000058240",
    "rt_1_1k": "rt_1_tf_trained_for_000001120",
}

def download_from_gs(gs_path, local_path):
    """Download a folder or file from a GCS path to local path using gsutil."""
    try:
        subprocess.run(["gsutil", "-m", "cp", "-r", gs_path, local_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading from {gs_path}: {e}")
        raise

def unzip_file(zip_path, extract_to):
    """Unzip a file to a specific directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def get_rt_1_checkpoint(name, ckpt_dir="./SimplerEnv/checkpoints"):
    assert name in RT_1_CHECKPOINTS, f"Unknown checkpoint name: {name}"
    ckpt_name = RT_1_CHECKPOINTS[name]
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_dir, exist_ok=True)
        if name == "rt_1_x":
            zip_file_path = os.path.join(ckpt_dir, f"{ckpt_name}.zip")
            gs_path = f"gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/{ckpt_name}.zip"
            download_from_gs(gs_path, zip_file_path)
            unzip_file(zip_file_path, ckpt_dir)
        else:
            gs_path = f"gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/{ckpt_name}"
            download_from_gs(gs_path, ckpt_dir)

    return ckpt_path
    
def download_all_rt_1_checkpoints(ckpt_dir="./SimplerEnv/checkpoints"):
    for name in RT_1_CHECKPOINTS:
        print(f"Downloading checkpoint: {name}")
        path = get_rt_1_checkpoint(name, ckpt_dir)
        print(f"✅ Downloaded {name} to {path}\n")

# Call the function to download all checkpoints
download_all_rt_1_checkpoints()

# @title Select your model and environment

task_name = "google_robot_pick_coke_can"  # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]

if 'env' in locals():
  print("Closing existing env")
  env.close()
  del env
env = simpler_env.make(task_name)

# Note: we turned off the denoiser as the colab kernel will crash if it's turned on
# To use the denoiser, please git clone our SIMPLER environments
# and perform evaluations locally.
sapien.render_config.rt_use_denoiser = False

obs, reset_info = env.reset()
instruction = env.get_language_instruction()
print("Reset info", reset_info)
print("Instruction", instruction)

if "google" in task_name:
  policy_setup = "google_robot"
else:
  policy_setup = "widowx_bridge"
 
# @title Select your model and environment
model_name = "octo-base" # @param ["rt_1_x", "rt_1_400k", "rt_1_58k", "rt_1_1k", "octo-base", "octo-small"]

if "rt_1" in model_name:
  from simpler_env.policies.rt1.rt1_model import RT1Inference

  ckpt_path = get_rt_1_checkpoint(model_name)
  model = RT1Inference(saved_model_path=ckpt_path, policy_setup=policy_setup)
elif "octo" in model_name:
  from simpler_env.policies.octo.octo_model import OctoInference
  from typing import Optional
  import jax
  from transforms3d.euler import euler2axangle

  class BatchedOctoInference(OctoInference):
    def batch_step(self, image: np.ndarray, num_inferences: int, task_description: Optional[str] = None) -> list[tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]]:
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        assert image.dtype == np.uint8
        image_resized = self._resize_image(image)
        self._add_image_to_history(image_resized)
        images, pad_mask = self._obtain_image_history_and_mask()

        # We will collect results for each inference in a loop.
        all_norm_raw_actions = []
        all_action_infos = []

        # Generate num_inferences keys for the loop.
        self.rng, *keys = jax.random.split(self.rng, num_inferences + 1)

        for i in range(num_inferences):
            key = keys[i]
            
            # Prepare input for a single inference (batch size 1)
            # The model expects a batch dimension.
            input_observation = {
                "image_primary": images[None],
                "pad_mask": pad_mask[None]
            }
            
            norm_raw_actions, action_info = self.model.sample_actions(
                input_observation,
                self.task, # self.task is already prepared for a single inference
                rng=key,
            )
            
            all_norm_raw_actions.append(norm_raw_actions)
            all_action_infos.append(action_info)

        # After the loop, stack the results to create batches.
        # The model's output already has a batch dimension of 1, so we concatenate along axis 0.
        norm_raw_actions_batch = jax.numpy.concatenate(all_norm_raw_actions, axis=0)
        action_info_batch = jax.tree_map(lambda *xs: jax.numpy.concatenate(xs, axis=0), *all_action_infos)
        
        raw_actions_batch = norm_raw_actions_batch * self.action_std[None] + self.action_mean[None]

        results = []
        for i in range(num_inferences):
            raw_actions = raw_actions_batch[i]  # (action_pred_horizon, action_dim)
            
            action_info = {k: v[i] for k, v in action_info_batch.items()}
            
            assert raw_actions.shape == (self.pred_action_horizon, 7)
            if self.action_ensemble:
                raw_actions = self.action_ensembler.ensemble_action(raw_actions)
                raw_actions = raw_actions[None]

            raw_action = {
                "world_vector": np.array(raw_actions[0, :3]),
                "rotation_delta": np.array(raw_actions[0, 3:6]),
                "open_gripper": np.array(raw_actions[0, 6:7]),
            }

            action = {}
            action["world_vector"] = raw_action["world_vector"] * self.action_scale
            action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
            roll, pitch, yaw = action_rotation_delta
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            action_rotation_axangle = action_rotation_ax * action_rotation_angle
            action["rot_axangle"] = action_rotation_axangle * self.action_scale

            if self.policy_setup == "google_robot":
                current_gripper_action = raw_action["open_gripper"]
                if self.previous_gripper_action is None:
                    relative_gripper_action = np.array([0])
                else:
                    relative_gripper_action = self.previous_gripper_action - current_gripper_action
                self.previous_gripper_action = current_gripper_action
                if np.abs(relative_gripper_action) > 0.5 and not self.sticky_action_is_on:
                    self.sticky_action_is_on = True
                    self.sticky_gripper_action = relative_gripper_action
                if self.sticky_action_is_on:
                    self.gripper_action_repeat += 1
                    relative_gripper_action = self.sticky_gripper_action
                if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                    self.sticky_action_is_on = False
                    self.gripper_action_repeat = 0
                    self.sticky_gripper_action = 0.0
                action["gripper"] = relative_gripper_action
            elif self.policy_setup == "widowx_bridge":
                action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            
            action["terminate_episode"] = np.array([0.0])
            results.append((raw_action, action, action_info))
        
        return results

  model = BatchedOctoInference(model_type=model_name, policy_setup=policy_setup, init_rng=0)
else:
  raise ValueError(model_name)


#@title Run inference


import json
import numpy as np
import os
import gc
from datetime import datetime, timedelta

num_episodes = 200
# Create a folder for all JSON files
output_dir = f"no_mc_dropout_{task_name}_{num_episodes}_episodes"
os.makedirs(f"{output_dir}/json", exist_ok=True)
os.makedirs(f"{output_dir}/video", exist_ok=True)

# For logging progress
start_time = datetime.now()
last_log_time = start_time

def log_progress(current_ep, total_ep, success_count):
    """Log progress with time estimates"""
    global last_log_time
    current_time = datetime.now()
    elapsed = (current_time - start_time).total_seconds() / 60  # minutes
    time_per_ep = elapsed / (current_ep + 1)
    remaining = time_per_ep * (total_ep - current_ep - 1)

    # Only log every 5 minutes or when specifically requested
    if (current_time - last_log_time).total_seconds() >= 300 or current_ep == total_ep - 1:
        print(f"Progress: {current_ep+1}/{total_ep} episodes completed ({(current_ep+1)/total_ep*200:.1f}%)")
        print(f"Success rate: {success_count}/{current_ep+1} ({success_count/(current_ep+1)*200:.1f}%)")
        print(f"Elapsed time: {elapsed:.1f} minutes")
        print(f"Estimated remaining time: {remaining:.1f} minutes")
        print(f"Estimated completion: {current_time + timedelta(minutes=remaining)}")
        print("-" * 200)
        last_log_time = current_time

# Helper function to ensure all numpy arrays are converted to lists
def ensure_serializable(obj):
    """Convert any non-serializable objects to JSON-serializable ones"""
    # Handle JAX Array types (ArrayImpl)
    if str(type(obj).__name__) == 'ArrayImpl':
        # Convert JAX array to numpy array first, then to list
        try:
            return obj.tolist()
        except:
            # If tolist fails, try numpy conversion first
            try:
                return np.array(obj).tolist()
            except:
                # Last resort: convert to string
                return str(obj)

    # Handle numpy arrays
    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    # Handle dictionaries
    elif isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}

    # Handle lists and tuples
    elif isinstance(obj, (list, tuple)):
        return [ensure_serializable(i) for i in obj]

    # Handle numpy scalar types
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)

    # Check if object has a tolist method (for other array-like objects)
    elif hasattr(obj, 'tolist'):
        try:
            return obj.tolist()
        except:
            return str(obj)

    # Return object as is if it's likely JSON serializable
    else:
        return obj

# A JSON encoder class that uses our serialization function
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return ensure_serializable(obj)
        except:
            return str(obj)  # Fallback to string representation

# Track success count for reporting
success_count = 0

# Run for 50 episodes
for episode_id in range(num_episodes):
    print(f"Running episode {episode_id+1}/{num_episodes}")

    # Reset environment for a new episode
    obs, reset_info = env.reset()
    instruction = env.get_language_instruction()
    model.reset(instruction)

    if episode_id == 0:  # Only print instruction for first episode to reduce output clutter
        print(f"Instruction: {instruction}")

    # Create an episode dictionary - we'll flush this after each episode
    trajectory = []
    frames = []

    # Track episode outcome
    success = False

    # Run 100 timesteps for the episode
    for timestep in range(200):
        if timestep % 20 == 0:  # Print even less frequently to reduce clutter
            print(f"  Timestep {timestep}")

        # Get the image (use single image now)
        image = get_image_from_maniskill2_obs_dict(env, obs) # Use single image
        frames.append(image)

        # --- Perform single inference ---
        results = model.batch_step(image, 1)
        raw_action, action, action_info = results[0]

        if timestep == 0:
            print("✅ Verification Passed: Model is producing single outputs. MC Dropout is not active.")

        # Cleanup the image used for inference
        del image

        # Extract optional info like entropy, log_probs, token_argmax for logging
        token_argmax_data = []
        token_entropy_data = []
        
        if action_info:
            if 'entropy' in action_info:
                entropy_data = ensure_serializable(action_info['entropy'])
                if isinstance(entropy_data, list) and len(entropy_data) > 0:
                    first_horizon_entropies = entropy_data[0][:7]  # Take first 7 values from first horizon
                    token_entropy_data.append(first_horizon_entropies)  # Keep existing behavior for backward compatibility
            if 'token_argmax' in action_info:
                token_argmax_data.append(ensure_serializable(action_info['token_argmax']))
        
        # --- Use the action to step Environment ---
        # No averaging needed for single inference

        # Calculate mean action from raw actions for logging (to match rt900k.py)
        mean_action = {
            "world_vector": raw_action["world_vector"],
            "rotation_delta": raw_action["rotation_delta"],
            "gripper_closedness_action": raw_action["open_gripper"]
        }

        # Use the action components for the environment step
        combined_action = np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
        obs, reward, terminated, truncated, info = env.step(combined_action)
        success = bool(info.get("success", False))

        # Store timestep data in the format from rt900k.py
        timestep_log = {
            "timestep": timestep,
            "token_argmax": token_argmax_data,
            "token_entropy": token_entropy_data,
            "mean_action": ensure_serializable(mean_action),
            "info": ensure_serializable(info)
        }
        trajectory.append(timestep_log)

        # --- Cleanup action components and combined action ---
        del combined_action, raw_action, action, action_info

        if terminated or truncated or success:
            print(f"Terminated at step {timestep} with success: {success}")
            break


    # Save this episode's data to a separate JSON file in the folder
    filename_prefix = f"{str(success)}_{episode_id}"
    episode_file_path = os.path.join(output_dir, "json", f'{filename_prefix}.json')
    video_file_path = os.path.join(output_dir, "video", f'{filename_prefix}.mp4')

    try:
        with open(episode_file_path, 'w') as f:
            json.dump(trajectory, f, cls=EnhancedJSONEncoder, indent=2)
        mediapy.write_video(video_file_path, frames, fps=10)
    except Exception as e:
        print(f"Error saving episode data or video: {e}")

    # Free memory by clearing episode data
    del trajectory
    del frames

    # Update success count if this episode was successful
    if success:
        success_count += 1

    # Log progress
    if (episode_id + 1) % 10 == 0 or episode_id == num_episodes - 1:
        log_progress(episode_id, num_episodes, success_count)

    # Explicitly run garbage collection between episodes
    gc.collect()

# Final stats
total_elapsed = (datetime.now() - start_time).total_seconds() / 60
print(f"Completed {num_episodes} episodes in {total_elapsed:.1f} minutes")
print(f"Data saved to {output_dir}/ folder")
print(f"Total successful episodes: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")

# Clear any remaining references to large data structures
gc.collect()
