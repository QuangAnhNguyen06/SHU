import os
import numpy as np
from PIL import Image
import tensorflow as tf
import random

# Load folder & file
def load_file_pairs(folder_path):
    """
    Load all file pairs (.jpg and .npy) from the specified folder.
    Each pair is matched by the same name (e.g., 0.jpg and 0.npy).

    Parameters:
        folder_path (str): Path to the folder containing .jpg and .npy files.

    Returns:
        list: A list of tuples, where each tuple contains paths to a .jpg file and its corresponding .npy file.
    """
    # Get all files in the folder
    all_files = os.listdir(folder_path)

    # Separate .jpg and .npy files
    jpg_files = {os.path.splitext(f)[0]: f for f in all_files if f.endswith('.jpg')}
    npy_files = {os.path.splitext(f)[0]: f for f in all_files if f.endswith('.npy')}

    # Find matching pairs by filename
    pairs = []
    for name in jpg_files:
        if name in npy_files:
            pairs.append((os.path.join(folder_path, jpg_files[name]),
                          os.path.join(folder_path, npy_files[name])))
    return pairs


# Read .jpg files
def read_jpg_files(image_path):
    """
    # Read file .jpg
    image_arr: ndarray
    """  
    try:
        
        image_arr = Image.open(image_path)
        image_arr = np.array(image_arr)
    except Exception as e:
        print(f"Error image: {e}")
        image_arr = None
    
    return image_arr

# Read .npy files
def read_and_parse_npy_file(file_path):
    """
    # Read file .npy
    tip_array: list
    joints_array: list
    """
    data = np.load(file_path, allow_pickle=True).item()
    tip_array = data.get('tip', [])[0] if 'tip' in data and data['tip'] else []
    joints_array = data.get('joints', [])[0] if 'joints' in data and data['joints'] else []

    return tip_array + joints_array


# Combine .jpg and .npy
def combine_data(image, npy_vector):
    """
    # Combine file .npy and .jpg
    combined: float32
    """
    image = tf.expand_dims(image, axis=-1)  # (H, W, 1)
    image = tf.cast(image, tf.float32)
    npy_vector = tf.cast(npy_vector, tf.float32)  # float32
    combined = tf.concat([tf.reshape(image, [-1]), npy_vector], axis=0)
    return combined







##### Barlow Twin
def off_diagonal(x):
    n = tf.shape(x)[0]
    flattened = tf.reshape(x, [-1])[:-1]
    off_diagonals = tf.reshape(flattened, (n-1, n+1))[:, 1:]
    return tf.reshape(off_diagonals, [-1])


def normalize_repr(z):
    z_norm = (z - tf.reduce_mean(z, axis=0)) / tf.math.reduce_std(z, axis=0)
    return z_norm


def compute_loss(z_a, z_b, lambd):
    # Get batch size and representation dimension.
    batch_size = tf.cast(tf.shape(z_a)[0], z_a.dtype)
    repr_dim = tf.shape(z_a)[1]

    # Normalize the representations along the batch dimension.
    z_a_norm = normalize_repr(z_a)
    z_b_norm = normalize_repr(z_b)

    # Cross-correlation matrix.
    c = tf.matmul(z_a_norm, z_b_norm, transpose_a=True) / batch_size

    # Loss.
    on_diag = tf.linalg.diag_part(c) + (-1)
    on_diag = tf.reduce_sum(tf.pow(on_diag, 2))
    off_diag = off_diagonal(c)
    off_diag = tf.reduce_sum(tf.pow(off_diag, 2))
    loss = on_diag + (lambd * off_diag)
    return loss   







class BarlowTwins(tf.keras.Model):
    def __init__(self, encoder, lambd=5e-3):
        super(BarlowTwins, self).__init__()
        self.encoder = encoder
        self.lambd = lambd
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        # Unpack the data.
        ds_one, ds_two = data

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z_a, z_b = self.encoder(ds_one, training=True), self.encoder(ds_two, training=True)
            loss = compute_loss(z_a, z_b, self.lambd) 

        # Compute gradients and update the parameters.
        gradients = tape.gradient(loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()} 
    







# Combined model for training
class Combined_BarlowTwins(tf.keras.Model):
    def __init__(self, network_1, network_2, lambd=5e-3, lambda_image=1.0, lambda_joint=1.0):
        super(Combined_BarlowTwins, self).__init__()
        self.network_1 = network_1
        self.network_2 = network_2
        self.lambd = lambd
        self.lambda_image = lambda_image
        self.lambda_joint = lambda_joint
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        # Unpack the data: (image_1, image_2), (joint_1, joint_2)
        img_aug_1, img_aug_2, joint_aug_1, joint_aug_2 = data

        # Forward pass for both networks
        with tf.GradientTape() as tape:
            z_img_1, z_img_2 = self.network_1(img_aug_1), self.network_1(img_aug_2)
            z_joint_1, z_joint_2 = self.network_2(joint_aug_1), self.network_2(joint_aug_2)
            # Loss
            loss_img = compute_loss(z_img_1, z_img_2, self.lambd)
            loss_joint = compute_loss(z_joint_1, z_joint_2, self.lambd)
            # Combine losses
            loss_total = self.lambda_image * loss_img + self.lambda_joint * loss_joint

        # Compute gradients and update parameters for both networks
        trainable_vars = self.network_1.trainable_variables + self.network_2.trainable_variables
        gradients = tape.gradient(loss_total, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Monitor loss
        self.loss_tracker.update_state(loss_total)
        return {"loss": self.loss_tracker.result()}


























### Create dataset
def split_and_shuffle_pairs(shape, folder_path, seed=226, split_number=500):
    """
    Tạo ra 4 biến cấu trúc từ đầu vào theo yêu cầu.

    Args:
        shape (list of tuples): Danh sách các cặp đường dẫn (.jpg, .npy).
        folder_path (str): Đường dẫn thư mục chứa các file.
        seed (int): Seed để đảm bảo tính ngẫu nhiên cố định.

    Returns:
        tuple: Gồm 4 danh sách:
            - 500 cặp ngẫu nhiên, tương đồng.
            - 500 cặp ngẫu nhiên, không tương đồng.
            - Còn lại, tương đồng.
            - Còn lại, không tương đồng.
    """
    # Đảm bảo tính ngẫu nhiên cố định
    random.seed(seed)

    # Copy và shuffle dữ liệu ban đầu
    all_pairs = shape.copy()
    random.shuffle(all_pairs)

    # Chọn 500 cặp ngẫu nhiên tương đồng
    random_n_similar = all_pairs[:split_number]

    # Tạo 500 cặp không tương đồng
    random_n_dissimilar = []
    for jpg, _ in random_n_similar:
        # Chọn ngẫu nhiên một file .npy không phải cặp gốc
        while True:
            random_npy = random.choice(random_n_similar)[1]
            if random_npy != _:
                random_n_dissimilar.append((jpg, random_npy))
                break

    # Tách phần còn lại
    remaining_pairs = all_pairs[split_number:]

    # Chia thành 2 nhóm: tương đồng và không tương đồng
    remaining_similar = remaining_pairs
    remaining_dissimilar = []
    for jpg, _ in remaining_similar:
        # Chọn ngẫu nhiên một file .npy không phải cặp gốc
        while True:
            random_npy = random.choice(remaining_similar)[1]
            if random_npy != _:
                remaining_dissimilar.append((jpg, random_npy))
                break

    return random_n_similar, random_n_dissimilar, remaining_similar, remaining_dissimilar