import tensorflow as tf

from multiprocessing import cpu_count


class DataBase(object):
    def __init__(self, training=False, batch_size=64):
        train, test = tf.keras.datasets.mnist.load_data()
        data = train if training else test
        self.img, self.label, self.iter, self.iter_init = self.get_imgs_and_labels(data, batch_size, training)
        tf.add_to_collection("Iterator_init", self.iter_init)

    def get_imgs_and_labels(self, data, batch_size, training):
        # Convert numpy data to tf.dataset
        with tf.name_scope('train_dataset' if training else 'test_dataset') as scope:
            dataset = self.create_dataset(data, batch_size)
            iterator = dataset.make_initializable_iterator()
        return list(iterator.get_next()) + [iterator, iterator.initializer]

    def create_dataset(self, data, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.repeat().shuffle(buffer_size=batch_size * 10)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(batch_size * 10)
        dataset = dataset.map(self._parse_function, num_parallel_calls=cpu_count())
        return dataset

    def _parse_function(self, img, label):
        img = self._img_preprocessing(img)
        label = self._label_preprocessing(label)
        return img, label

    def _img_preprocessing(self, img):
        pp_img = tf.to_float(img) / 255.
        return pp_img

    def _label_preprocessing(self, label):
        pp_label = tf.to_int64(label, name='label')
        return pp_label

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