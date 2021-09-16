import tensorflow as tf

class CRNN(tf.keras.Model):
    """Implementation of the Convolutional Recurrent Neural Network
    described in the Primus paper.
    """

    def __init__(self, alphabet_size: int):
        """
        Args:
            alphabet_size: size of the alphabet of the music representaiton.
        """
        super().__init__()
        self.alphabet_size=alphabet_size
        # Conv block
        conv_1 = tf.keras.layers.Conv2D(
            filters = 32,
            kernel_size = 3, 
            activation='relu', 
            data_format = 'channels_last',
            input_shape=(128, None, 1)
        )
        conv_2 = tf.keras.layers.Conv2D(
            filters = 64,
            kernel_size = 3, 
            activation='relu', 
            data_format = 'channels_last'
        )
        conv_3 = tf.keras.layers.Conv2D(
            filters = 128,
            kernel_size = 3, 
            activation='relu', 
            data_format = 'channels_last'
        )
        conv_3 = tf.keras.layers.Conv2D(
            filters = 128,
            kernel_size = 3, 
            activation='relu', 
            data_format = 'channels_last'
        )
        conv_4 = tf.keras.layers.Conv2D(
            filters = 256,
            kernel_size = 3, 
            activation='relu', 
            data_format = 'channels_last'
        )
        max_pool = tf.keras.layers.MaxPooling2D(data_format='channels_last')
        batch_norm = tf.keras.layers.BatchNormalization
        layers = [
            conv_1, batch_norm(), max_pool,
            conv_2, batch_norm(), max_pool,
            conv_3, batch_norm(), max_pool,
            conv_4, batch_norm(), max_pool
        ]
        self.conv_block = tf.keras.Sequential(layers = layers)
        # RNN block
        blstm_1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(256, return_sequences=True),
            merge_mode = 'ave'
        )
        blstm_2 = tf.keras.layers.Bidirectional(
           tf.keras.layers.LSTM(256, return_sequences=True),
           merge_mode='ave'
        )
        self.rnn_block = tf.keras.Sequential(
            layers = [blstm_1, blstm_2]
        )
        self.dense = tf.keras.layers.Dense(self.alphabet_size+1, activation='softmax') 

    def call(self, images: tf.Tensor, training: bool):
        """Pass images through conv, rnn, and dense blocks.

        Args:
            images: a batch of images.
        """
        #print('\nimage shape: ', images.shape)
        out = self.conv_block(images, training)
        #print('conv out shape: ', out.shape)
        # shape is [batch, reduced_height, width, channels].
        # transpose to [batch, width, channels, height]
        # because reshape maintains order, so each feature
        # map column will be contiguous, and feature map
        # columns will be concatenated.
        out = tf.transpose(out, perm = [0, 2, 3, 1])
        #print('transpose out shape: ', out.shape)
        # reshape to [batch, width, channels*height]
        out = tf.reshape(out, [out.shape[0], out.shape[1], -1])
        #print('reshape out shape: ', out.shape)
        out = self.rnn_block(out)
        #print('rnn out shape: ', out.shape)
        out = self.dense(out)
        return out
    
    def compile(self, optimizer, **compile_kwargs):
        """Compiles with CTC Loss.

        Args:
            optimizer: optimizer to be used for training.
            compile_kwargs: arguments for tf.keras.Model.compile()
        """
        super().compile(**compile_kwargs)
        self.optim = optimizer
        self.loss = tf.nn.ctc_loss

    def test_step(self, data):
        images, sequences, seq_lens = data
        logits = self.call(images, training=False)
        logits = tf.transpose(logits, perm = [1, 0, 2])
        logit_len = logits.shape[0]
        batch_size = logits.shape[1]
        logit_length = tf.constant([logit_len]*batch_size, dtype=tf.dtypes.int32)
        loss = self.loss(
            labels = sequences,
            logits = logits,
            label_length = seq_lens,
            logit_length = logit_length,
            blank_index = -1
        )
        preds, _ = tf.nn.ctc_greedy_decoder(logits, logit_length)
        unpacked_preds = tf.sparse.to_dense(preds[0])
        return {'loss': loss}


    def train_step(self, data):
        images, sequences, seq_lens = data
        with tf.GradientTape() as tape:
            logits = self.call(images, training=True)
            logits = tf.transpose(logits, perm=[1,0,2])
            logit_len = logits.shape[0]
            batch_size = logits.shape[1]
            logit_length = tf.constant([logit_len]*batch_size, dtype=tf.dtypes.int32)
            loss = self.loss(
                labels = sequences, 
                logits = logits, 
                label_length = seq_lens, 
                logit_length = logit_length,
                blank_index = -1
            )
        grads = tape.gradient(loss, self.trainable_weights)
        self.optim.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': loss}
