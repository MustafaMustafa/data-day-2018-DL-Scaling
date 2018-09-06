import tensorflow as tf

class DatasetIteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(DatasetIteratorInitializerHook, self).__init__()

        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""

        if self.iterator_initializer_func is not None:
            self.iterator_initializer_func(session)
        else:
            print("DataIteratorInitializerHook: iterator intitalization function"
                  "is not set for this data iterator")
