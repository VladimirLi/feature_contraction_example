import tensorflow_datasets as tfds
import tensorflow as tf


def convert_types(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return image, label


def get_data(batch_size):
    """
    Returns dataset object of MNIST with batch size of batch_size
    :param batch_size: the batch size of the mini-batch
    :return: mnist_train, mnist_test datasets
    """
    # Getting MNIST dataset
    dataset, info = tfds.load('mnist', with_info=True, as_supervised=True)

    mnist_train, mnist_test = dataset['train'], dataset['test']
    mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(batch_size)
    mnist_test = mnist_test.map(convert_types).batch(batch_size)
    return mnist_train, mnist_test


