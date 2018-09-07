__author__ = "Mustafa Mustafa"
__email__  = "mmustafa@lbl.gov"

import tensorflow as tf
import h5py
import numpy as np
from data.iterator_initializer_hook import DatasetIteratorInitializerHook

def shuffle(a, b, seed):
    rand_state = np.random.RandomState(seed)
    rand_state.shuffle(a)
    rand_state.seed(seed)
    rand_state.shuffle(b)

def get_input_fn(filename, dataset_size, batchsize, epochs, variable_scope,
                 shuffle_buffer_size=12800, rank=None):
    """ creates a tf.data.Dataset and feeds and augments data from an h5 file

    Returns:
        data input function input_fn
        """

    with h5py.File(filename, mode='r', driver='core') as _f:
        data_group = _f['all_events']
        features = np.expand_dims(data_group['hist'][:dataset_size], axis=-1).astype(np.float32)
        labels = np.expand_dims(data_group['y'][:dataset_size], axis=-1).astype(np.float32)
        if rank is not None:
            shuffle(features, labels, seed=107+rank)
        _f.close()

    iterator_initializer_hook = DatasetIteratorInitializerHook()

    def input_fn():
        """ create input_fn for Estimator training

        Returns:
            tf.Tensors of features and labels
        """

        with tf.variable_scope(variable_scope) as _:
            features_placeholder = tf.placeholder(tf.float32, features.shape)
            labels_placeholder = tf.placeholder(tf.float32, labels.shape)

            dataset = tf.data.Dataset.from_tensor_slices((features_placeholder,
                                                          labels_placeholder))
            dataset = dataset.shuffle(shuffle_buffer_size)
            dataset = dataset.repeat(epochs)
            dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batchsize))
            dataset = dataset.prefetch(4)

            data_it = dataset.make_initializable_iterator()
            iterator_initializer_hook.iterator_initializer_func = \
                    lambda sess: sess.run(data_it.initializer,
                                          feed_dict={features_placeholder: features,
                                                     labels_placeholder: labels})
            X, y = data_it.get_next()

        return X, y

    return input_fn, iterator_initializer_hook
