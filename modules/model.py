from tensorflow.keras import Model
from tensorflow.keras import layers, regularizers


class FeatureContractionLayer(layers.Layer):
    def __init__(
            self,
            activation,
            *args,
            **kwargs):
        # TODO: change to former definition of feature contraction
        self.dense = layers.Dense(*args, **kwargs)
        self.activation = activation
        super(FeatureContractionLayer, self).__init__(name=kwargs["name"])

    def __call__(self, input_):
        net = self.dense(input_)
        return self.activation(net), net


class MnistModel(Model):
    def __init__(
            self,
            weight_decay=0,
            kernel_size=5,
            fc3_number_filters=1024,
            dropout_rate=0,
            name="MnistModel",
            batch_size=32,
            feature_contraction_weight=0,
            layer_to_contract=None,
    ):
        super(MnistModel, self).__init__(name=name)

        kernel_regularizer = None
        activity_regularizer = {"fc3": None, "fc4": None}
        self.dropout = layers.Dropout(dropout_rate) if dropout_rate != 0 else None
        if weight_decay != 0:
            kernel_regularizer = regularizers.l2(weight_decay)
        # if layer_to_contract:
        #     num_filters = 3136.0 if layer_to_contract == "fc3" else 1024.0
        #     feature_contraction_weight /= float(2*batch_size*num_filters)
        #     activity_regularizer[layer_to_contract] = \
        #         regularizers.l2(feature_contraction_weight)

        self.conv1 = layers.Conv2D(
            filters=32,
            kernel_size=kernel_size,
            kernel_regularizer=kernel_regularizer,
            padding="same",
            activation="relu",
            name="conv1"
        )
        self.pool1 = layers.MaxPooling2D(
            pool_size=2,
            strides=2,
            padding='valid',
            name="pool1"
        )
        self.conv2 = layers.Conv2D(
            filters=64,
            kernel_size=kernel_size,
            kernel_regularizer=kernel_regularizer,
            padding="same",
            activation="relu",
            name="conv2"
        )
        self.pool2 = layers.MaxPooling2D(
            pool_size=2,
            strides=2,
            padding='valid',
            name="pool2"
        )
        self.flatten = layers.Flatten()
        self.fc3 = FeatureContractionLayer(
            layers.Activation("relu"),
            fc3_number_filters,
            use_bias=True,
            kernel_regularizer=kernel_regularizer,
            name="fc3",
        )
        self.fc4 = FeatureContractionLayer(
            layers.Activation("softmax"),
            10,
            use_bias=True,
            kernel_regularizer=kernel_regularizer,
            name="fc4",
        )

    def call(self, input_, training=True):
        end_points = {}
        net = self.conv1(input_)
        end_points["conv1"] = net
        net = self.pool1(net)
        end_points["pool1"] = net
        net = self.conv2(net)
        end_points["conv2"] = net
        net = self.pool2(net)
        net = self.flatten(net)
        end_points["pool2"] = net
        net, _ = self.fc3(net)
        end_points["contract"] = _
        end_points["fc3"] = net
        if self.dropout:
            net = self.dropout(net, training=training)
        net, _ = self.fc4(net)
        return net, end_points


