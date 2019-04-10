import tensorflow as tf


class Experiment:
    def __init__(self, batch_size_, epochs_, feature_contraction_weight, *args, **kwargs):
        from modules import mnist
        from modules.model import MnistModel
        self.contraction_weight = feature_contraction_weight
        self.batch_size = batch_size_
        self.epochs = epochs_

        self.test_accuracy_list = []
        self.test_loss_list = []

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)

        self.model = MnistModel(*args, **kwargs)
        self.mnist_train, self.mnist_test = mnist.get_data(self.batch_size)

    @tf.function
    def train_step(self, image, label):
        with tf.GradientTape() as tape:
            predictions, _ = self.model(image)
            contract = _["contract"]

            loss = self.loss_object(label, predictions)
            if self.contraction_weight:
                loss += tf.nn.l2_loss(contract)/float(2*tf.math.reduce_prod(tf.shape(contract)))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(label, predictions)

    @tf.function
    def test_step(self, image, label):
        predictions, _ = self.model(image, training=False)
        t_loss = self.loss_object(label, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(label, predictions)

    def train_loop(self):
        for image, label in self.mnist_train:
            self.train_step(image, label)

    def test_loop(self):
        for test_image, test_label in self.mnist_test:
            self.test_step(test_image, test_label)

    def reset_all_states(self):
        self.train_accuracy.reset_states()
        self.test_accuracy.reset_states()
        self.train_loss.reset_states()
        self.test_loss.reset_states()

    def __call__(self):

        for epoch in range(self.epochs):

            self.train_loop()
            self.test_loop()
            # print("Epoch: {}\tacc: {}".format(epoch+1, self.test_accuracy.result()))

            self.test_loss_list.append(self.test_loss.result().numpy())
            self.test_accuracy_list.append(self.test_accuracy.result().numpy())
            self.reset_all_states()

