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
        self.id = uuid.uuid4()
        self.is_training = tf.placeholder_with_default(True, shape=())

    def embed(self, img):
        # Create an embedding based on given image
        flattened = tf.layers.flatten(img)
        layer_1 = tf.layers.dense(flattened, self.embed_dim, activation=tf.nn.relu)
        layer_1_nl = tf.layers.dropout(layer_1, 0.8, training=self.is_training)
        img_embed = tf.layers.dense(layer_1_nl, self.num_classes)

        # Check that the shapes are as we would expect
        assert img.shape[1:] == self.img_size
        assert flattened.shape[1:] == self.img_size[0] * self.img_size[1]
        assert img_embed.shape[1:] == self.num_classes

        return img_embed, layer_1_nl


class DataConfig(object):
    def __init__(self, batch_size=64):
        self.init_op = []
        self.batch_size = batch_size
        self.train_iter_init = None
        self.test_iter_init = None
        self.next_element = None
        self.create_data_init()

    def create_data_init(self):
        # Loading processed MNIST train and test data
        train, test = [self.preprocess_data(data) for data in tf.keras.datasets.mnist.load_data()]
        assert train[0].shape[1:] == test[0].shape[1:] == (28, 28)

        # Convert numpy data to tf.dataset
        train_dataset = self.create_dataset(train, batch_size=self.batch_size)
        test_dataset = self.create_dataset(test, batch_size=self.batch_size)

        # Creating and saving iterator and data inits
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        self.train_iter_init = iterator.make_initializer(train_dataset, name='Train')
        self.test_iter_init = iterator.make_initializer(test_dataset, name='Test')
        self.next_element = iterator.get_next()

    def preprocess_data(self, data):
        assert type(data) == tuple
        img, lbl = data
        img = img.astype(np.float32) / 255.
        lbl = lbl.astype(np.int64)
        return img, lbl

    def create_dataset(self, data, batch_size=None):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.repeat().shuffle(buffer_size=batch_size * 1000)
        dataset = dataset.batch(batch_size) if batch_size else dataset
        dataset = dataset.prefetch(batch_size * 4) if batch_size else dataset
        return dataset


class TrainLoss(object):
    def __init__(self):
        self.data_config = DataConfig()
        self.model_params = ModelParams()

    def eval(self):
        # Loss will be on the negative log likelihood that the img embed belongs to the correct class of numbers
        img, lbl = self.data_config.next_element
        logits, layer_1 = self.get_logits(img)

        # Determine the loss and probability of positive sample
        log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=lbl))
        pos_probability = tf.exp(-log_loss)
        assert log_loss.shape == pos_probability.shape == ()

        # Get the accuracy of prediction from logits compared to the label
        prediction = tf.argmax(logits, -1)
        accuracy = tf.reduce_mean(tf.to_float(tf.equal(prediction, lbl)))

        return log_loss, pos_probability, accuracy, prediction, lbl

    def get_logits(self, img):
        img_embed, layer_1 = self.model_params.embed(img)
        # logits = tf.matmul(img_embed, self.model_params.lbl_embed, transpose_b=True)
        # assert logits.shape[1:] == self.model_params.num_classes
        return img_embed, layer_1


class TrainRun(object):
    def __init__(self, sess, lr=0.001):
        self.train_loss = TrainLoss()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        id = self.train_loss.model_params.id
        self.train_writer = tf.summary.FileWriter('./.test_MNIST/{name}/train'.format(name=id), tf.get_default_graph())
        self.test_writer = tf.summary.FileWriter('./.test_MNIST/{name}/test'.format(name=id), tf.get_default_graph())
        self.tags = ['Log_Loss', 'Pos_Prob', 'Accuracy']
        self.eval_metrics = self.train_loss.eval()
        self.log_loss, self.pos_prob, self.accuracy, self.pred, self.lbl = self.eval_metrics
        self.train_op = self.optimizer.minimize(self.log_loss)
        self.initialize(sess)
        self.step = 0

    def initialize(self, sess):
        print('Initializing Values: \n{init_vals}'.format(init_vals=sess.run(tf.report_uninitialized_variables())))
        init_op = [tf.global_variables_initializer(), self.train_loss.data_config.train_iter_init]
        sess.run(init_op)
        print('Finished Initialization.')

    def train_step(self, sess):
        for step in range(60 * 10**3):
            _ = sess.run([self.train_op])
            self.step += 1
            if step % 10**3 == 0:
                print('Evaluating metrics...')
                self.report_metrics(sess)

    def report_metrics(self, sess):
        # Evaluate using training data here
        train_loss, train_pos_prob, train_accuracy, train_pred, train_lbl = sess.run(
            self.eval_metrics, feed_dict={self.train_loss.model_params.is_training: False})

        # Write train metrics to tensorboard
        for tag, value in zip(self.tags, (train_loss, train_pos_prob, train_accuracy)):
            self.tensorboard_logger(self.train_writer, tag=tag, value=value)

        # Switch to testing data and get metrics
        sess.run(self.train_loss.data_config.test_iter_init)
        test_loss, test_pos_prob, test_accuracy, test_pred, test_lbl = sess.run(
            self.eval_metrics, feed_dict={self.train_loss.model_params.is_training: False})

        # Write test metrics to tensorboard
        for tag, value in zip(self.tags, [test_loss, test_pos_prob, test_accuracy]):
            self.tensorboard_logger(self.test_writer, tag=tag, value=value)

        # Print out eval metrics for comparison
        print('Train loss: {train:.3f} \t\t\t Test loss: {test:.3f}'.format(
            train=train_loss, test=test_loss))
        print('Train Pos Prob: {train:.3f} \t\t\t Test Pos Prob: {test:.3f}'.format(
            train=train_pos_prob, test=test_pos_prob))
        print('Train Accuracy: {train:.3f} \t\t\t Test Accuracy: {test:.3f}'.format(
            train=train_accuracy, test=test_accuracy))
        print('Train Predict {train} \t Test Predict {test}'.format(
            train=train_pred[:10], test=test_pred[:10]))
        print('Train Correct {train} \t Test Correct {test}'.format(
            train=train_lbl[:10], test=test_lbl[:10]))

        # Switch data back to training data
        sess.run(self.train_loss.data_config.train_iter_init)

    def tensorboard_logger(self, writer, tag, value):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        writer.add_summary(summary, self.step)
        writer.flush()


if __name__ == '__main__':
    sess = tf.Session()
    tr = TrainRun(sess)
    tr.train_step(sess)
