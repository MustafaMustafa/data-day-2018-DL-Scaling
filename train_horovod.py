import os
import tensorflow as tf
import horovod.tensorflow as hvd
from models.cnn_model import CNN_Model
from data.data_pipeline import get_input_fn
from hparams.yparams import YParams
from config_device import config_device

def model_fn(features, labels, params, mode):
    """ Build graph and return EstimatorSpec """

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    params.learning_rate = hvd.size() * params.learning_rate
    model = CNN_Model(params, input_x=features, is_training=is_training)

    if mode is not tf.estimator.ModeKeys.PREDICT:
        # loss and optimizer are not needed for inference
        model.define_loss(labels)
        model.define_optimizer()
        model.optimizer = hvd.DistributedOptimizer(model.optimizer)
        model.define_train_op()

    with tf.variable_scope('eval_metrics') as _:
        eval_metrics = {}
        probs = tf.nn.sigmoid(model.logits)
        predictions = tf.round(probs)
        eval_metrics['acc'] = tf.metrics.accuracy(labels=labels, predictions=predictions)

    return tf.estimator.EstimatorSpec(loss=model.loss,
                                      train_op=model.train_op,
                                      eval_metric_ops=eval_metrics,
                                      mode=mode)
def main(argv):
    """ Training loop """

    if len(argv) != 3:
        print("Usage", argv[0], "configuration_YAML_file", "configuration")
        exit()

    hvd.init()
    rank0 = (hvd.rank() == 0)
    if rank0:
        print("Number of horovod ranks =", hvd.size())

    # load hyperparameters
    params = YParams(os.path.abspath(argv[1]), argv[2], print_params=rank0)
    params.experiment_dir = params.experiment_dir + '_' + str(hvd.size()) + '_nodes'

    # create training data input pipeline
    train_input_fn, train_init_hook = get_input_fn(params.train_data_files,
                                                   dataset_size=params.train_dataset_size,
                                                   batchsize=params.batchsize,
                                                   epochs=params.epochs,
                                                   variable_scope='train_data_pipeline')

    # # create validation data input pipeline
    valid_input_fn, valid_init_hook = get_input_fn(params.valid_data_files,
                                                   dataset_size=params.valid_dataset_size,
                                                   batchsize=params.batchsize,
                                                   epochs=params.epochs,
                                                   variable_scope='valid_data_pipeline')

    # build estimator
    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    model_dir = params.experiment_dir if rank0 else None

    session_config = config_device('KNL')
    config = tf.estimator.RunConfig(session_config=session_config,
                                    save_checkpoints_secs=200 if rank0 else None) # TOO FREQUENT, JUST FOR DEMO, use defaults instead
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=model_dir,
                                       config=config,
                                       params=params)

    # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states from
    # rank 0 to all other processes. This is necessary to ensure consistent
    # initialization of all workers when training is started with random weights or
    # restored from a checkpoint.
    bcast_hook = hvd.BroadcastGlobalVariablesHook(0)

    tf.logging.set_verbosity(tf.logging.INFO)

    steps_per_epoch = (params.train_dataset_size//params.batchsize)
    for epoch in range(params.epochs):

        estimator.train(input_fn=train_input_fn,
                        hooks=[train_init_hook, bcast_hook],
                        max_steps=(epoch+1)*steps_per_epoch)

        if rank0:
            estimator.evaluate(input_fn=valid_input_fn,
                               steps=params.valid_dataset_size//params.batchsize,
                               hooks=[valid_init_hook])

if __name__ == '__main__':
    tf.app.run()
