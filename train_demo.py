import os
import tensorflow as tf
from models.cnn_model import CNN_Model
from data.data_pipeline import get_input_fn
from hparams.yparams import YParams
from config_device import config_device

def model_fn(features, labels, params, mode):
    """ Build graph and return EstimatorSpec """

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = CNN_Model(params, input_x=features, is_training=is_training)

    if mode is not tf.estimator.ModeKeys.PREDICT:
        # loss and optimizer are not needed for inference
        model.define_loss(labels)
        model.define_optimizer()
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

    # load hyperparameters
    params = YParams(os.path.abspath(argv[1]), argv[2])

    # build estimator
    session_config = config_device('KNL')
    config = tf.estimator.RunConfig(session_config=session_config,
                                    save_checkpoints_secs=200) # TOO FREQUENT, JUST FOR DEMO, use defaults instead
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=params.experiment_dir,
                                       config=config,
                                       params=params)

    # create training data input pipeline
    train_input_fn, train_init_hook = get_input_fn(params.train_data_files,
                                                   dataset_size=params.train_dataset_size,
                                                   batchsize=params.batchsize,
                                                   epochs=params.epochs,
                                                   variable_scope='train_data_pipeline')

    max_steps = (params.train_dataset_size//params.batchsize)*params.epochs
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        hooks=[train_init_hook],
                                        max_steps=max_steps)

    # create validation data input pipeline
    valid_input_fn, valid_init_hook = get_input_fn(params.valid_data_files,
                                                   dataset_size=params.valid_dataset_size,
                                                   batchsize=params.batchsize,
                                                   epochs=params.epochs,
                                                   variable_scope='valid_data_pipeline')

    eval_spec = tf.estimator.EvalSpec(input_fn=valid_input_fn,
                                      hooks=[valid_init_hook])

    # train
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
    tf.app.run()
