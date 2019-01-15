import tensorflow as tf

from multiprocessing import cpu_count


class DataBase(object):
    def __init__(self, training=False, batch_size=64):
        train, test = tf.keras.datasets.mnist.load_data()
        data = train if training else test
        self.img, self.label, self.iter_init = self.get_imgs_and_labels(data, batch_size)

    def get_imgs_and_labels(self, data, batch_size):
        # Convert numpy data to tf.dataset
        dataset = self.create_dataset(data, batch_size)
        iterator = dataset.make_initializable_iterator()
        return list(iterator.get_next()) + [iterator.initializer]

    def create_dataset(self, data, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.repeat().shuffle(buffer_size=batch_size * 10)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(batch_size * 10)
        dataset = dataset.map(self._parse_function, num_parallel_calls=cpu_count())
        return dataset

    def _parse_function(self, img, label):
        img = tf.to_float(img) / 255.
        label = tf.to_int64(label)
        return img, label


class TrainData(DataBase):
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        super(TrainData, self).__init__(training=True)


class TestData(DataBase):
    def __init__(self, batch_size=1024):
        self.batch_size = batch_size
        super(TestData, self).__init__(batch_size=batch_size, training=False)



if __name__ == '__main__':
    data_base = DataBase()
    assert data_base.img.shape[1:] == (28, 28)
    assert data_base.label.shape[1:] == ()

    train_data = TrainData()
    assert train_data.img.shape[1:] == (28, 28)
    assert train_data.label.shape[1:] == ()

    test_data = TestData()
    assert test_data.img.shape[1:] == (28, 28)
    assert test_data.label.shape[1:] == ()

    test = tf.placeholder(dtype=tf.float32, shape=(train_data.img.shape))
    test1 = test * 2.

    with tf.Session() as sess:
        sess.run(test_data.iter_init)
        img, label = sess.run([test_data.img, test_data.label])
        print(img.shape, label.shape)

        output = sess.run([test1], feed_dict={test: train_data.img})