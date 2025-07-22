import tensorflow as tf
import numpy as np
import trimesh
import os

def _parse_mesh_graph_nets_tfexample(example_proto):
    """Parses a single tf.Example from the MeshGraphNets dataset."""
    feature_description = {
        'nodes': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'edges': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'senders': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'receivers': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'globals': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'mesh_pos': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'world_pos': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'node_type': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'velocity': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'pressure': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'cells': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'target_drag_coefficient': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        'target_max_von_mises_stress': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    }
    return tf.io.parse_single_example(example_proto, feature_description)

def load_mesh_graph_nets_dataset(tfrecord_path, task_type='airfoil'):
    """
    Loads and parses a MeshGraphNets TFRecord dataset.

    Args:
        tfrecord_path (str): Path to the TFRecord file.
        task_type (str): 'airfoil' for drag coefficient or 'elastic' for max von Mises stress.

    Returns:
        tf.data.Dataset: A dataset of parsed examples.
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = dataset.map(_parse_mesh_graph_nets_tfexample)

    def _extract_relevant_features(parsed_example):
        mesh_pos = parsed_example['mesh_pos']
        cells = parsed_example['cells']

        # Reshape mesh_pos to (num_vertices, 3)
        num_vertices = tf.shape(mesh_pos)[0] // 3
        mesh_pos = tf.reshape(mesh_pos, (num_vertices, 3))

        # Reshape cells to (num_faces, 3) for triangular meshes
        num_cells = tf.shape(cells)[0] // 3
        cells = tf.reshape(cells, (num_cells, 3))

        if task_type == 'airfoil':
            target_property = parsed_example['target_drag_coefficient']
        elif task_type == 'elastic':
            target_property = parsed_example['target_max_von_mises_stress']
        else:
            raise ValueError("task_type must be 'airfoil' or 'elastic'")

        return mesh_pos, cells, target_property

    return parsed_dataset.map(_extract_relevant_features)

def sample_points_from_mesh(vertices, faces, num_points=2048):
    """
    Samples points from a mesh surface and computes normals.

    Args:
        vertices (np.ndarray): Mesh vertices (N, 3).
        faces (np.ndarray): Mesh faces (M, 3).
        num_points (int): Number of points to sample.

    Returns:
        tuple: (sampled_points, sampled_normals)
    """
    if not isinstance(vertices, np.ndarray):
        vertices = vertices.numpy()
    if not isinstance(faces, np.ndarray):
        faces = faces.numpy()

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    sampled_points, face_indices = mesh.sample(num_points, return_index=True)
    sampled_normals = mesh.face_normals[face_indices]
    return sampled_points, sampled_normals

def normalize_point_cloud(points):
    """Normalizes a point cloud to fit within a unit sphere."""
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    points = points / max_dist
    return points

def generate_sdf(mesh_vertices, mesh_faces, query_points):
    """
    Generates Signed Distance Function (SDF) values for query points.

    Args:
        mesh_vertices (np.ndarray): Mesh vertices (N, 3).
        mesh_faces (np.ndarray): Mesh faces (M, 3).
        query_points (np.ndarray): Query points (K, 3).

    Returns:
        np.ndarray: SDF values for query points (K,).
    """
    if not isinstance(mesh_vertices, np.ndarray):
        mesh_vertices = mesh_vertices.numpy()
    if not isinstance(mesh_faces, np.ndarray):
        mesh_faces = mesh_faces.numpy()
    if not isinstance(query_points, np.ndarray):
        query_points = query_points.numpy()

    mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
    sdf_values = trimesh.proximity.signed_distance(mesh, query_points)
    return sdf_values

if __name__ == '__main__':
    # Example Usage (assuming you have downloaded the dataset)
    # You need to download the dataset first, e.g., from
    # https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
    # and place the TFRecord files in data/deepmind-research/meshgraphnets/datasets/

    # For elastic deformation (flag_simple)
    elastic_tfrecord_path = os.path.join(
        os.getcwd(), 'data', 'deepmind-research', 'meshgraphnets', 'datasets',
        'flag_simple', 'train.tfrecord'
    )
    if os.path.exists(elastic_tfrecord_path):
        print(f"Loading elastic dataset from: {elastic_tfrecord_path}")
        elastic_dataset = load_mesh_graph_nets_dataset(elastic_tfrecord_path, task_type='elastic')
        for i, (mesh_pos, cells, max_stress) in enumerate(elastic_dataset.take(1)):
            print(f"Elastic Example {i+1}:")
            print(f"  Mesh Vertices Shape: {mesh_pos.shape}")
            print(f"  Mesh Faces Shape: {cells.shape}")
            print(f"  Max Von Mises Stress: {max_stress.numpy()}")

            # Example of sampling points and generating SDF
            sampled_points, sampled_normals = sample_points_from_mesh(mesh_pos, cells)
            print(f"  Sampled Points Shape: {sampled_points.shape}")
            print(f"  Sampled Normals Shape: {sampled_normals.shape}")

            normalized_points = normalize_point_cloud(sampled_points)
            print(f"  Normalized Sampled Points Shape: {normalized_points.shape}")

            # Generate some random query points for SDF
            query_points = np.random.rand(100, 3) * 2 - 1 # Points in [-1, 1] cube
            sdf_values = generate_sdf(mesh_pos, cells, query_points)
            print(f"  SDF Values Shape: {sdf_values.shape}")
            print(f"  First 5 SDF values: {sdf_values[:5]}")
    else:
        print(f"Elastic TFRecord not found at: {elastic_tfrecord_path}")
        print("Please download the MeshGraphNets dataset and place it in the correct location.")
