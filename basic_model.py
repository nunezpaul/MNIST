import tensorflow as tf


class ModelParams(object):
    def __init__(self):
        self.img_size = (28, 28)
        self.batch_size = 64
        self.embed_dim = 256
        self.num_classes = 10
        self.lbl_embed = tf.Variable(tf.random_normal((self.num_classes, self.embed_dim)))

    # Create embedding based on given image
    def embed(self, img):
        assert img.shape[1:] == self.img_size
        img /= 255  # normalize the input to be from 0 to 1
        img -=0.5  # normalize the input to be from -0.5 to 0.5
        flattened = tf.reshape(img, (-1, self.img_size[0] * self.img_size[1]))
        assert flattened.shape[1:] == self.img_size[0] * self.img_size[1]
        layer_1 = tf.layers.dense(flattened, self.embed_dim)
        img_embed = tf.nn.tanh(tf.layers.batch_normalization(layer_1))
        assert layer_1.shape[1:] == img_embed.shape[1:] == (self.embed_dim)
        return img_embed


class TrainLoss(object):
    def __init__(self, sess):
        self.model_params = ModelParams()
        self.init_op = []
        self.session = sess

    def get_iter(self):
        train, test = tf.keras.datasets.mnist.load_data()
        assert train[0].shape[1:] == test[0].shape[1:] == (28, 28)
        train_iter = self.convert_data(train, batch_size=self.model_params.batch_size)
        test_iter = self.convert_data(test, batch_size=self.model_params.batch_size * 4)
        return train_iter, test_iter

    def convert_data(self, data, batch_size=None):
        assert type(data) == tuple
        img = tf.to_float(data[0])
        lbl = tf.to_int32(data[1])
        dataset = tf.data.Dataset.from_tensor_slices((img, lbl))
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size) if batch_size else dataset
        dataset = dataset.prefetch(batch_size * 4) if batch_size else dataset
        iterator = dataset.make_initializable_iterator()
        self.init_op.append(iterator.initializer)
        return iterator

    def get_logits(self, data_iterator):
        img, lbl = data_iterator.get_next()
        img_embed = self.model_params.embed(img)
        logits = tf.matmul(img_embed, self.model_params.lbl_embed, transpose_b=True)
        assert logits.shape[1:] == self.model_params.num_classes
        return logits, lbl

    def loss(self):
        # Loss will be on the negative log likelihood that the img embed belongs to the correct class of numbers
        train_iter, _ = self.get_iter()
        logits, lbl = self.get_logits(train_iter)
        neg_log_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=lbl)
        neg_log_loss_avg = tf.reduce_mean(neg_log_loss)
        train_accuracy = tf.exp(-neg_log_loss_avg)
        assert neg_log_loss_avg.shape == train_accuracy.shape == ()
        tf.summary.scalar('Train_Accuracy', train_accuracy)
        tf.summary.scalar('Negative_Log_Loss', neg_log_loss_avg)
        return neg_log_loss_avg

    def eval_loss(self):
        _, test_iter = self.get_iter()
        logits, lbl = self.get_logits(test_iter)
        prediction = tf.argmax(logits, -1)
        correct_pred = tf.equal(tf.to_int32(prediction), lbl)
        test_loss = 1 - tf.reduce_mean(tf.to_float(correct_pred))
        assert test_loss.shape == ()
        tf.summary.scalar('Test_loss', test_loss)
        return test_loss


class TrainRun(object):
    def __init__(self, sess, lr=0.1):
        self.train_loss = TrainLoss(sess)
        self.optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
        self.summary_writer = tf.summary.FileWriter('./.test_MNIST/test', sess.graph)
        self.loss = self.train_loss.loss()
        self.test_loss = self.train_loss.eval_loss()
        self.train_op = self.optimizer.minimize(self.loss)
        self.initialize(sess)
        self.counter = 0

    def initialize(self, sess):
        print('Initializing Values.')
        init_op = [tf.global_variables_initializer()]
        sess.run(init_op + self.train_loss.init_op)
        print('Finished Initialization.')

    def train_step(self, sess):
        for step in range(10**3):
            _ = sess.run([self.train_op])
            self.counter += 1
            if step % 10**2 == 0:
                train_loss, test_loss = sess.run([self.loss, self.test_loss])
                print('Train loss: {train} \t Test loss: {test}'.format(train=train_loss, test=test_loss))
                _ = self.recorder()

    def recorder(self):
        merge = tf.summary.merge_all()
        summary = sess.run(merge)
        self.summary_writer.add_summary(summary, self.counter)


if __name__ == '__main__':
    sess = tf.Session()
    tr = TrainRun(sess)
    tr.train_step(sess)
