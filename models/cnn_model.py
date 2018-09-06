import numpy as np
import tensorflow as tf

# pylint: disable-msg=C0103
# pylint: disable=too-many-instance-attributes

class CNN_Model(object):

    def __init__(self, params, input_x=None, is_training=True):

        self._params = params
        self._is_training = is_training
        self._data_format = params.data_format
        self.input_x = input_x

        with tf.variable_scope('training_counters', reuse=tf.AUTO_REUSE) as _:
            self.global_step = tf.train.get_or_create_global_step()

        self.build_graph()
        self.loss = None
        self.optimizer = None
        self.train_op = None

    def build_graph(self):
        """ network """

        # Convolutional layers
        _h = self.input_x

        with tf.variable_scope('model'):
            for _fs in self._params.conv_filters:
                _h = tf.layers.conv2d(_h, filters=_fs, **self._params.conv_args)
                _h = tf.layers.max_pooling2d(_h, **self._params.maxpool_args)

            if self._params.conv_dropout_rate:
                _h = tf.layers.dropout(_h, rate=self._params.conv_dropout_rate,
                                       training=self._is_training)
            _h = tf.layers.flatten(_h)

            # Fully connected  layers
            for _u, _do in zip(self._params.fc_hidden_units, self._params.fc_dropout_rates):
                _h = tf.layers.dense(_h, units=_u, activation=self._params.fc_activation)
                if _do:
                    _h = tf.layers.dropout(_h, rate=_do, training=self._is_training)

            # Ouptut layer
            self.logits = tf.layers.dense(_h, units=1)

        print("Number of model parameters", np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))

    def define_loss(self, labels):
        """ define loss """

        with tf.name_scope('sigmoid_cross_entropy'):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=self.logits))

    def define_optimizer(self):
        """ build optimizer """

        with tf.variable_scope('optimizer') as _:
            self.optimizer = tf.train.AdamOptimizer(self._params.learning_rate)

    def define_train_op(self):
        """ build train_op """

        with tf.variable_scope('train_op') as _:
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
