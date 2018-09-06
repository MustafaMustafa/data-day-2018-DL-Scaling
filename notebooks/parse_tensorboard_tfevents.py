import os
import glob
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
tf.logging.set_verbosity(tf.logging.ERROR)

def parse_tfevents_file(filepath):

    history = {}
    for event in tf.train.summary_iterator(filepath):
        for value in event.summary.value:

            if value.tag not in history:
                history[value.tag] = {'wall_time': [], 'steps': [], 'values': []}

            if value.HasField('simple_value'):
                history[value.tag]['wall_time'].append(event.wall_time)
                history[value.tag]['steps'].append(event.step)
                history[value.tag]['values'].append(value.simple_value)

    for key in history.keys():

        history[key]['wall_time'] = np.array(history[key]['wall_time'])
        history[key]['steps'] =     np.array(history[key]['steps'])
        history[key]['values'] =    np.array(history[key]['values'])

        #subtract start time
        if len(history[key]['wall_time']):
            history[key]['time'] = np.array(history[key]['wall_time']) - history[key]['wall_time'][0]

    return history

def parse_tfevents_dir(_dir):

    event_acc = EventAccumulator(_dir)
    event_acc.Reload()

    history = {}
    for tag in event_acc.Tags()['scalars']:
        history[tag] = {'wall_time': [], 'steps': [], 'values': []}
        w_times, steps, values = zip(*event_acc.Scalars(tag))
        history[tag]['wall_time'] = np.array(list(w_times))
        history[tag]['steps'] = np.array(list(steps))
        history[tag]['values'] = np.array(list(values))

        #subtract start time
        if len(history[tag]['wall_time']):
            history[tag]['time'] = np.array(history[tag]['wall_time']) - history[tag]['wall_time'][0]

    return history

def get_training_history(model_dir):
    train_hist = parse_tfevents_dir(model_dir)
    valid_hist = parse_tfevents_dir(model_dir+'/eval/')

    return train_hist, valid_hist
