import tensorflow as tf

class SequenceAccuracy(tf.keras.metrics.Metric):
    """ Calculates equality of entire prediction sequence to entire label sequence
    """
    def __init__(self, name='sequence_accuracy', **kwargs):
        super().__init__(**kwargs)
        self.correct = self.add_weight(name='correct_count', initializer='zeros')
        self.total = self.add_weight(name='total_count', initializer='zeros')

    def update_state(self, labels, preds, sample_weight=None):
        """Update the state of the metric."""
        # remove blanks
        mask = tf.not_equal(preds, tf.constant(0, dtype=preds.dtype))
        preds = tf.ragged.boolean_mask(preds, mask)
        #pad to target sequence lenghts
        preds = preds.to_tensor(default_value=-1, shape=labels.shape)
        sequence_acc = tf.equal(labels, preds)
        # tensor where equality gives 1, otherwise 0
        sequence_acc = tf.where(
            sequence_acc, 
            tf.fill(tf.shape(sequence_acc), 1),
            tf.fill(tf.shape(sequence_acc), 0)
        )
        sequence_acc = tf.reduce_min(sequence_acc, axis=1)
        batch_true = tf.reduce_sum(sequence_acc)
        batch_size = tf.squeeze(tf.shape(sequence_acc))
        self.correct.assign_add(tf.cast(batch_true, tf.float32))
        self.total.assign_add(tf.cast(batch_size, tf.float32))

    def result(self):
        """ Return the rate of correct sequence predictions."""
        return self.correct/self.total

    def reset_states(self):
        self.correct.assign(0.)
        self.total.assign(0.)

    


