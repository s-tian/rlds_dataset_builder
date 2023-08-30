from typing import Iterator, Tuple, Any

import pickle
from PIL import Image
import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class StanfordMaskVit(tfds.core.GeneratorBasedBuilder):

    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(480, 480, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(15,),
                            dtype=np.float32,
                            doc='Robot state, consists of [7x robot joint angles, 7x robot joint velocities,'
                                '1x gripper position].',
                        ),
                        'end_effector_pose': tfds.features.Tensor(
                            shape=(5,),
                            dtype=np.float32,
                            doc='Robot end effector pose, consists of [3x Cartesian position, '
                                '1x gripper yaw, 1x gripper position]. This is the state used in the MaskViT paper.',
                        ),
                        'finger_sensors': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float32,
                            doc='1x Sawyer gripper finger sensors.',
                        ),
                        'high_bound': tfds.features.Tensor(
                            shape=(5,),
                            dtype=np.float32,
                            doc='High bound for end effector pose normalization. '
                                'Consists of [3x Cartesian position, 1x gripper yaw, 1x gripper position].',
                        ),
                        'low_bound': tfds.features.Tensor(
                            shape=(5,),
                            dtype=np.float32,
                            doc='Low bound for end effector pose normalization. '
                                'Consists of [3x Cartesian position, 1x gripper yaw, 1x gripper position].',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(5,),
                        dtype=np.float32,
                        doc='Robot action, consists of [3x change in end effector position, '
                            '1x gripper yaw, 1x open/close gripper (-1 means to open the gripper, 1 means close)].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(paths=
                                             [('/svl/u/stian/tars_09_16/train',
                                              '/viscam/u/stian/mfm/data/mfm_robot_data_09_16_meta/train_trajectories.txt'),
                                              ('/svl/u/stian/mfm_backup/tars/train',
                                               '/viscam/u/stian/mfm/data/mfm_robot_data_meta/train_trajectories.txt')]),
            'val': self._generate_examples(paths=
                                             [('/svl/u/stian/tars_09_16/train',
                                               '/viscam/u/stian/mfm/data/mfm_robot_data_09_16_meta/val_trajectories.txt'),
                                              ('/svl/u/stian/mfm_backup/tars/train',
                                               '/viscam/u/stian/mfm/data/mfm_robot_data_meta/val_trajectories.txt')]),
        }

    def _generate_examples(self, paths) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        language_instruction = "push something"
        # compute Kona language embedding
        language_embedding = self._embed([language_instruction])[0].numpy()
        infer_gripper = True

        def _parse_example(traj_path):
            # load raw data --> this should change for your dataset
            cam_path = os.path.join(traj_path, "images0")
            image_paths = glob.glob(os.path.join(cam_path, "*.jpg"))
            # The 3: index below slices an image which has name in the format ex. im_6.jpg and slices off
            # the "im" part.
            image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0][3:]))
            action_path = os.path.join(traj_path, "policy_out.pkl")
            obs_path = os.path.join(traj_path, "obs_dict.pkl")

            with open(action_path, "rb") as f:
                per_step_policy_log = pickle.load(f, encoding="latin1")
                actions = np.stack([t["actions"] for t in per_step_policy_log])
            if infer_gripper:
                # Infer gripper action like in
                # https://github.com/SudeepDasari/visual_foresight/blob/4c79886cf6a01f2dba19fec8638073702eaf7ef9/visual_mpc/utils/file_2_record.py#L39
                with open(obs_path, "rb") as f:
                    obs_dict = pickle.load(f, encoding="latin1")
                    gripper_actions = []
                    for i in range(len(actions)):
                        if obs_dict["state"][i + 1, -1] >= 0.5:
                            gripper_actions.append(np.array([-1.0]))
                        else:
                            gripper_actions.append(np.array([1.0]))
                    high_bound = obs_dict["high_bound"].astype(np.float32)
                    low_bound = obs_dict["low_bound"].astype(np.float32)
                    finger_sensors = np.array(obs_dict["finger_sensors"]).astype(np.float32)[:, None]
                    qpos_qvel = np.concatenate((obs_dict["qpos"], obs_dict["qvel"]), axis=-1)
                    states = np.concatenate((qpos_qvel, obs_dict["state"][:, -1:]), axis=-1).astype(np.float32)
                    eep = obs_dict["state"][:].astype(np.float32)
                gripper_actions = np.stack(gripper_actions).astype(actions.dtype)
                actions = np.concatenate((actions, gripper_actions), axis=-1)
            else:
                actions = np.concatenate(
                    (actions, np.zeros_like(actions[:, -1])[..., None]), axis=-1
                )
            # Append dummy action to not break sampling, but this action should not be used!
            dummy_action = np.zeros(actions.shape[-1])
            actions = np.concatenate((actions, dummy_action[None]), axis=0).astype(np.float32)
            # load images into np.uint8 array
            images = []
            for image_path in image_paths:
                with Image.open(image_path) as image:
                    image_np = np.array(image)
                    images.append(image_np)

            # # assemble episode --> here we're not assuming demos so we set reward to 0 always
            episode = []
            for i, image in enumerate(images):
                episode.append({
                    'observation': {
                        'image': image,
                        # 'wrist_image': step['wrist_image'],
                        'state': states[i],
                        'end_effector_pose': eep[i],
                        'finger_sensors': finger_sensors[i],
                        'low_bound': low_bound[i],
                        'high_bound': high_bound[i],
                    },
                    'action': actions[i],
                    'discount': 1.0,
                    'reward': 0,
                    'is_first': i == 0,
                    'is_last': i == (len(images) - 1),
                    'is_terminal': i == (len(images) - 1),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': traj_path
                }
            }
            # if you want to skip an example for whatever reason, simply return None
            return traj_path, sample

        episode_paths = []
        # create list of all examples
        for path, meta_path in paths:
            with open(meta_path, 'r') as f:
                for line in f:
                    episode_paths.append(os.path.join(path, line.strip()))
        print("Found {} episodes".format(len(episode_paths)))

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )


if __name__ == "__main__":
    dataset = StanfordMaskVit()
    split_gens = dataset._split_generators(None)
    for split_name, split_gen in split_gens.items():
        print(split_name)
        for i, (key, value) in enumerate(split_gen):
            # print(key, value)
            breakpoint()
            if i > 1:
                break