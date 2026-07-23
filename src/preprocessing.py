"""Preprocessing functions for malaria cell images."""
import tensorflow as tf

IMG_SIZE = 128
BATCH_SIZE = 32

def load_and_preprocess(filepath, label):
    # read image file, decode, resize, normalize to 0-1 range
    image = tf.io.read_file(filepath)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = image / 255.0
    return image, label

def augment(image, label):
    # random flips and rotation, training set only
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    return image, label

def make_dataset(df, augment_data=False, shuffle=False):
    # builds a tf.data pipeline: load, batch, optionally shuffle and augment
    ds = tf.data.Dataset.from_tensor_slices((df["filepath"].values, df["label"].values))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=42)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def preprocess_single_image(image_path):
    # for single-image prediction via the API, adds batch dimension
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)
    return image