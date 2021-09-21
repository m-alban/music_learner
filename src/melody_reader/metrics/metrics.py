import tensorflow as tf

class SequenceAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='sequence_accuracy', **kwargs):
        super().__init__(**kwargs)
        self.true = self.add_weight(name='true_seq', initializer='zeros')
        self.total = self.add_weight(name='seq_count', initializer='zeros')

    def update_state(self, sequences, preds, sample_weight=None):        
        # remove blanks
        mask = tf.not_equal(preds, tf.constant(0, dtype=preds.dtype))
        preds = tf.ragged.boolean_mask(preds,mask)
        #pad to target sequence lenghts
        preds = preds.to_tensor(default_value=-1, shape=sequences.shape)
        sequence_acc = tf.equal(sequences, preds)
        # tensor where equality gives 1, otherwise 0
        sequence_acc = tf.where(
            sequence_acc, 
            tf.fill(tf.shape(sequence_acc), 1),
            tf.fill(tf.shape(sequence_acc), 0)
        )
        sequence_acc = tf.reduce_min(sequence_acc, axis=1)
        batch_true = tf.reduce_sum(sequence_acc)
        batch_size = tf.squeeze(tf.shape(sequence_acc))
        self.true.assign_add(tf.cast(batch_true, tf.float32))
        self.total.assign_add(tf.cast(batch_size, tf.float32))

    def result(self):
        return self.true/self.total

    def reset_states(self):
        self.true.assign(0.)
        self.total.assign(0.)

    


