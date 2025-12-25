import os
import numpy as np
from pathlib import Path
from fire import Fire
from loguru import logger
from natsort import natsorted

from datasets.preprocessing.base_preprocessing import BasePreprocessing


class WheatheadIISPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "data/tiled_data_3x3/",
        save_dir: str = "data/processed/wheathead_iis/",
        modes: tuple = (
            "9",
            "10",
            "11",
            "12",
        ),
        n_jobs: int = -1,
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)

        # Define your class mapping here
        # Adjust based on your semantic classes
        self.class_map = {
            "stem": 0,
            "leaves": 1,
            "wheat-heads": 2,
            # Add more classes as needed
        }

        

        self.create_label_database()

        # Assuming your data is organized as:
        # data_dir/mode/scene_name/
        #   - points.txt (N x 4: x, y, z, intensity)
        #   - instance_labels.txt (N x 1: instance_id, 0=ignore)
        #   - semantic_labels.txt (N x 1: semantic_class_id)
        for mode in self.modes:
            filepaths = []
            mode_dir = self.data_dir / mode
            if mode_dir.exists():

                for scene_path in [f.path for f in os.scandir(mode_dir) if f.name.endswith("_point.npy")]:
                    filepaths.append(scene_path)
            self.files[mode] = natsorted(filepaths)

    def create_label_database(self):
        label_database = dict()
        for class_name, class_id in self.class_map.items():
            label_database[class_id] = {
                "color": [255, 255, 255],  # Default white, you can customize
                "name": class_name,
                "validation": True,
            }

        self._save_yaml(self.save_dir / "label_database.yaml", label_database)
        return label_database

    def process_file(self, filepath, mode):
        """process_file.

        Args:
            filepath: path to the scene directory
            mode: train, test or validation

        Returns:
            filebase: info about file
        """
        filebase = {
            "filepath": filepath,
            "scene": Path(filepath).name,
            "area": mode,
            "raw_filepath": str(filepath),
            "file_len": -1,
        }

        scene_name = Path(filepath).name.replace("_point.npy", "")

        # Load the three files
        points_file = Path(filepath).parent / f"{scene_name}_point.npy"
        instance_file = Path(filepath).parent / f"{scene_name}_ins_label.npy"
        semantic_file = Path(filepath).parent / f"{scene_name}_sem_label.npy"

        if not all(f.exists() for f in [points_file, instance_file, semantic_file]):
            raise FileNotFoundError(f"Missing files in {filepath}")

        # Load data
        points = np.load(points_file)  # N x 4: x, y, z, intensity
        instance_ids = np.load(instance_file).astype(int)  # N x 1
        semantic_labels = np.load(semantic_file).astype(int)  # N x 1

        if not (len(points) == len(instance_ids) == len(semantic_labels)):
            raise ValueError(f"File lengths don't match in {filepath}")

        N = len(points)

        # Create the expected format:
        # [x, y, z, r, g, b, nx, ny, nz, segment_id, semantic_label, instance_label]

        # Coordinates
        coords = points[:, :3] / 1000.0  # Convert mm to meters

        # Intensity as grayscale color (you can modify this)
        colors = points[:, 3:]
        # colors = np.concatenate([intensity, intensity, intensity], axis=1)  # N x 3
        # Dummy normals (you can compute real normals if needed)
        normals = np.ones((N, 3))

        # Segment IDs - for now, use instance IDs as segment IDs
        # In Mask3D, segments are used for grouping points
        segment_ids = instance_ids.copy()

        # Semantic and instance labels
        semantic_labels_processed = semantic_labels.copy()
        instance_ids_processed = instance_ids.copy()

        # Handle ignore labels (0 in instance labels)
        # Convert 0 to -1 for ignore (following Mask3D convention)
        instance_ids_processed[instance_ids_processed == 0] = -99
        instance_ids_processed = instance_ids_processed - 1
        instance_ids_processed[instance_ids_processed == -100] = -1

        # Combine all data

        processed_points = np.hstack([
            coords,                    # 0-2: x, y, z
            colors,                    # 3 intensity as grayscale
            normals,                   # 4-6: nx, ny, nz
            segment_ids.reshape(-1, 1), # 7: segment_id
            semantic_labels_processed.reshape(-1, 1),  # 8: semantic label
            instance_ids_processed.reshape(-1, 1),   # 9: instance id
        ])

        filebase["file_len"] = N

        # Save processed data
        processed_filepath = self.save_dir / mode / f"{scene_name}.npy"
        processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, processed_points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        # Create instance ground truth
        # This is used for evaluation - format: (semantic_class + 1) * 1000 + instance_id + 1
        # Only for non-ignore instances
        valid_mask = instance_ids_processed >= 0
        if valid_mask.any():
            gt_data = (semantic_labels_processed + 1) * 1000 + instance_ids_processed + 1
        else:
            gt_data = np.array([])

        processed_gt_filepath = self.save_dir / "instance_gt" / mode / f"{scene_name}.txt"
        processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_data.astype(np.int32), fmt="%d")
        filebase["instance_gt_filepath"] = str(processed_gt_filepath)

        # Color stats (dummy values since we don't have real colors)
        filebase["color_mean"] = [0.5]  # Normalized intensity mean
        filebase["color_std"] = [0.5]   # Normalized intensity std

        return filebase

    def joint_database(
        self,
        train_modes=(
            "9",
            "10",
            "11",
            "12",
        ),
    ):
        for mode in train_modes:
            joint_db = []
            for let_out in train_modes:
                if mode == let_out:
                    continue
                joint_db.extend(
                    self._load_yaml(
                        self.save_dir / (let_out + "_database.yaml")
                    )
                )
            self._save_yaml(
                self.save_dir / f"train_{mode}_database.yaml", joint_db
            )


if __name__ == "__main__":
    Fire(WheatheadIISPreprocessing)
