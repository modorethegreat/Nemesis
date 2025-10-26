import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
import trimesh
import json
import os
from src.data_processing.mesh_data_extractor import sample_points_from_mesh, normalize_point_cloud, generate_sdf

# The following functions are adapted from the MeshGraphNets repository
# https://github.com/deepmind/deepmind-research/blob/master/meshgraphnets/common.py

def _parse_function(example_proto, meta):
    feature_lists = {k: tf.io.VarLenFeature(tf.string)
                   for k in meta['field_names']}
    features = tf.io.parse_single_example(example_proto, feature_lists)
    out = {}
    for key, field in meta['features'].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
        data = tf.reshape(data, field['shape'])
        if field['type'] == 'static':
            data = tf.tile(data, [meta['trajectory_length'], 1, 1])
        elif field['type'] == 'dynamic_varlen':
            length = tf.io.decode_raw(features['length_'+key].values, tf.int32)
            length = tf.reshape(length, [-1])
            ragged = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
            padding_value = field.get('padding_value', 0.0)
            data = ragged.to_tensor(default_value=padding_value)
            out[f'{key}_lengths'] = length
        elif field['type'] != 'dynamic':
            raise ValueError('invalid data format')
        out[key] = data
    return out

def _load_meta(tfrecord_path):
    meta_path = os.path.join(os.path.dirname(tfrecord_path), 'meta.json')
    with open(meta_path, 'r') as fp:
        meta = json.loads(fp.read())
    return meta


def _get_tfrecord_dataset(tfrecord_path, meta):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(lambda x: _parse_function(x, meta))
    return dataset


def _get_length_from_meta(meta):
    for key in ('num_samples', 'num_records', 'num_examples', 'count', 'dataset_length', 'length'):
        if key in meta:
            try:
                return int(meta[key])
            except (TypeError, ValueError):
                continue
    return None

def get_target_property(record):
    # For airfoil, use the average pressure at the last time step
    # pressure is (trajectory_length, num_nodes, 1)
    return np.mean(record['pressure'][-1])

class MeshDataset(Dataset):
    """Dataset loader that returns geometric samples and flow features.

    The ``__getitem__`` dictionary contains:

    * ``points``: ``(num_points, 3)`` surface samples normalised to the unit cube.
    * ``normals``: ``(num_points, 3)`` normals for the sampled points.
    * ``cells``: ``(F, 3)`` mesh connectivity used for SDF queries.
    * ``sdf_points``: ``(num_sdf_points, 3)`` query coordinates for the decoder.
    * ``sdf_values``: ``(num_sdf_points,)`` signed distances from the mesh surface.
    * ``label``: scalar target value for regression.
    * ``<flow_feature>``: ``(trajectory_length, max_nodes, feature_dim)`` dense
      tensors for each ``dynamic_varlen`` flow field (e.g. velocity or pressure).
    * ``<flow_feature>_lengths``: ``(trajectory_length,)`` integer tensors with
      the node count per step to recover the unpadded mesh trajectories.
    """

    def __init__(self, tfrecord_path, num_points=2048, num_sdf_points=1024, is_local=False, batch_size=1):
        self.tfrecord_path = tfrecord_path
        self.num_points = num_points
        self.num_sdf_points = num_sdf_points

        self.meta = _load_meta(tfrecord_path)
        base_dataset = _get_tfrecord_dataset(tfrecord_path, self.meta)

        full_length = _get_length_from_meta(self.meta)
        if full_length is None:
            cardinality = tf.data.experimental.cardinality(base_dataset)
            if cardinality == tf.data.experimental.UNKNOWN_CARDINALITY:
                # Materialise a count without keeping records in memory
                full_length = sum(1 for _ in base_dataset)
                base_dataset = _get_tfrecord_dataset(tfrecord_path, self.meta)
            else:
                full_length = int(cardinality.numpy())

        if is_local:
            requested = batch_size * 2
            if full_length is None:
                self.length = requested
                dataset = base_dataset.take(requested)
            else:
                self.length = min(full_length, requested)
                dataset = base_dataset.take(self.length)
        else:
            # Load the entire dataset into memory (for full training)
            self.dataset = list(get_tfrecord_iterator(tfrecord_path))
        self.length = len(self.dataset)
        self.dynamic_flow_keys = []
        if self.length:
            self.dynamic_flow_keys = sorted(
                key[:-len('_lengths')]
                for key in self.dataset[0].keys()
                if key.endswith('_lengths')
            )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError('Index out of range')

        dataset_slice = self._dataset.skip(idx).take(1)
        try:
            record = next(iter(dataset_slice))
        except StopIteration:
            raise IndexError('Index out of range')

        record = tf.nest.map_structure(lambda x: x.numpy() if hasattr(x, 'numpy') else x, record)

        if 'mesh_pos' in record:
            positions = record['mesh_pos'][-1]
        elif 'world_pos' in record:
            positions = record['world_pos'][-1]
        else:
            raise ValueError("Positions not found in record")
        cells = record['cells'][-1]

        # Ensure positions are 3D for trimesh
        if positions.shape[1] == 2:
            positions_for_trimesh = np.hstack([positions, np.zeros((positions.shape[0], 1))])
        else:
            positions_for_trimesh = positions

        mesh = trimesh.Trimesh(vertices=positions_for_trimesh, faces=cells)

        # Sample points from the mesh surface
        sampled_points, sampled_normals = sample_points_from_mesh(positions_for_trimesh, cells, self.num_points)
        normalized_points = normalize_point_cloud(sampled_points)

        # Sample points for SDF calculation
        query_points = np.random.rand(self.num_sdf_points, 3) * 2 - 1 # Points in [-1, 1] cube
        sdf_values = trimesh.proximity.signed_distance(mesh, query_points)

        # Get the label
        label = get_target_property(record)

        sample = {
            'points': torch.from_numpy(normalized_points.copy()).float(),
            'normals': torch.from_numpy(sampled_normals.copy()).float(),
            'cells': torch.from_numpy(cells.copy()).long(), # Add cells to the output
            'sdf_points': torch.from_numpy(query_points.copy()).float(),
            'sdf_values': torch.from_numpy(sdf_values.copy()).float(),
            'label': torch.from_numpy(np.array(label).copy()).float(),
        }

        for key in self.dynamic_flow_keys:
            flow = record[key]
            lengths = record[f'{key}_lengths']
            sample[key] = torch.from_numpy(flow.copy()).float()
            sample[f'{key}_lengths'] = torch.from_numpy(lengths.copy()).long()

        return sample

def collate_fn(batch):
    collated = {
        'points': torch.stack([item['points'] for item in batch]),
        'normals': torch.stack([item['normals'] for item in batch]),
        'cells': torch.stack([item['cells'] for item in batch]), # Add cells to the collated batch
        'sdf_points': torch.stack([item['sdf_points'] for item in batch]),
        'sdf_values': torch.stack([item['sdf_values'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
    }

    length_keys = [key for key in batch[0].keys() if key.endswith('_lengths')]
    for length_key in length_keys:
        feature_key = length_key[:-len('_lengths')]
        feature_tensors = [item[feature_key] for item in batch]
        max_time = max(tensor.shape[0] for tensor in feature_tensors)
        max_nodes = max(tensor.shape[1] for tensor in feature_tensors)

        padded_features = []
        padded_lengths = []
        length_tensors = [item[length_key] for item in batch]
        for tensor, lengths in zip(feature_tensors, length_tensors):
            pad_time = max_time - tensor.shape[0]
            pad_nodes = max_nodes - tensor.shape[1]
            padded_features.append(F.pad(tensor, (0, 0, 0, pad_nodes, 0, pad_time)))
            pad_len = max_time - lengths.shape[0]
            padded_lengths.append(F.pad(lengths, (0, pad_len)))

        collated[feature_key] = torch.stack(padded_features)
        collated[length_key] = torch.stack(padded_lengths)

    return collated
