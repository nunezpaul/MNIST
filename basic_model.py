import tensorflow as tf
import uuid
import numpy as np


class ModelParams(object):
    def __init__(self):
        self.img_size = (28, 28)
        self.batch_size = 64
        self.embed_dim = 256
        self.num_classes = 10
        self.lbl_embed = tf.Variable(tf.random_normal((self.num_classes, self.embed_dim)))
        self.name = uuid.uuid4()

    # Create embedding based on given image
    def embed(self, img, training=True):
        assert img.shape[1:] == self.img_size
        flattened = tf.layers.flatten(img)
        assert flattened.shape[1:] == self.img_size[0] * self.img_size[1]
        layer_1 = tf.layers.dense(flattened, self.embed_dim * 2, activation=tf.nn.relu)
        layer_1_nl = tf.layers.dropout(layer_1, 0.2) if training else layer_1
        img_embed = tf.layers.dense(layer_1_nl, self.num_classes)
        assert img_embed.shape[1:] == self.num_classes
        return img_embed, layer_1

class DataConfig(object):
    def __init__(self):
        self.model_params = ModelParams()
        self.init_op = []
        self.train_iter, self.test_iter = self.get_iter()

    def initialize(self, sess):
        sess.run(self.init_op)
        self.init_op = []

    def get_iter(self):
        train, test = tf.keras.datasets.mnist.load_data()
        assert train[0].shape[1:] == test[0].shape[1:] == (28, 28)
        train_iter = self.convert_data(train, batch_size=self.model_params.batch_size)
        test_iter = self.convert_data(test, batch_size=self.model_params.batch_size * 4)
        return train_iter, test_iter

    def convert_data(self, data, batch_size=None):
        assert type(data) == tuple
        img, lbl = data
        img = img.astype(np.float32) / 255.
        lbl = lbl.astype(np.int32)
        dataset = tf.data.Dataset.from_tensor_slices((img, lbl))
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size) if batch_size else dataset
        dataset = dataset.prefetch(batch_size * 4) if batch_size else dataset
        iterator = dataset.make_initializable_iterator()
        self.init_op.append(iterator.initializer)
        return iterator

class TrainLoss(object):
    def __init__(self):
        self.data_config = DataConfig()
        self.model_params = self.data_config.model_params

    def loss(self):
        # Loss will be on the negative log likelihood that the img embed belongs to the correct class of numbers
        logits, lbl, layer_1 = self.get_logits(self.data_config.train_iter)
        neg_log_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=lbl)
        neg_log_loss_avg = tf.reduce_mean(neg_log_loss)
        train_accuracy = tf.exp(-neg_log_loss_avg)
        assert neg_log_loss_avg.shape == train_accuracy.shape == ()
        tf.summary.scalar('Train_Accuracy_1', train_accuracy)
        tf.summary.scalar('Negative_Log_Loss', neg_log_loss_avg)
        return neg_log_loss_avg

    def eval_loss(self):
        logits, lbl, layer_1 = self.get_logits(self.data_config.test_iter, train=False)
        assert logits.shape[1:] == self.model_params.num_classes
        prediction = tf.argmax(logits, -1)
        print(prediction.shape)
        correct_pred = tf.equal(prediction, tf.to_int64(lbl))
        test_accuracy = tf.reduce_mean(tf.to_float(correct_pred))
        assert test_accuracy.shape == ()
        tf.summary.scalar('Test_Accuracy', test_accuracy)
        return test_accuracy, prediction, lbl

    def get_logits(self, data_iterator, train=True):
        img, lbl = data_iterator.get_next()
        img_embed, layer_1 = self.model_params.embed(img, train)
        # logits = tf.matmul(img_embed, self.model_params.lbl_embed, transpose_b=True)
        # assert logits.shape[1:] == self.model_params.num_classes
        return img_embed, lbl, layer_1


class TrainRun(object):
    def __init__(self, sess, lr=0.01):
        self.train_loss = TrainLoss()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.summary_writer = tf.summary.FileWriter(
            './.test_MNIST/{name}'.format(name=self.train_loss.model_params.name), sess.graph)
        self.loss = self.train_loss.loss()
        self.eval_metrics = self.train_loss.eval_loss()
        self.train_op = self.optimizer.minimize(self.loss)
        self.initialize(sess)
        self.counter = 0

    def initialize(self, sess):
        print('Initializing Values.')
        init_op = [tf.global_variables_initializer()]
        sess.run(init_op + self.train_loss.data_config.init_op)
        print('Finished Initialization.')

    def train_step(self, sess):
        for step in range(60 * 10**3):
            _ = sess.run([self.train_op])
            self.counter += 1
            if step % 10**3 == 0:
                self.report_metrics(sess)

    def report_metrics(self, sess):
        train_loss, eval_metrics = sess.run([self.loss, self.eval_metrics])
        test_loss, prediction, correct_lbl = eval_metrics
        print('Train loss: {train} \t Test Accuracy: {test}'.format(train=train_loss, test=test_loss))
        print('predict {predict}'.format(predict=prediction[:10]))
        print('correct {correct}'.format(correct=correct_lbl[:10]))
        self.recorder(sess)

    def recorder(self, sess):
        merge = tf.summary.merge_all()
        summary = sess.run(merge)
        self.summary_writer.add_summary(summary, self.counter)


if __name__ == '__main__':
    sess = tf.Session()
    tr = TrainRun(sess)
    tr.train_step(sess)
