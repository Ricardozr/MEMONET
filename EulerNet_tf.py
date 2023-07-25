import tensorflow as tf
import numpy as np

class EulerInteractionLayer(object):
    def __init__(self, config, inshape, outshape):
        self.inshape, self.outshape = inshape, outshape
        self.feature_dim = config.embedding_size
        self.apply_norm = config.apply_norm

        # Initial assignment of the order vectors, which significantly affects the training effectiveness of the model.
        # We empirically provide two effective initialization methods here.
        # How to better initialize is still a topic to be further explored.
        # Note: 👆 👆 👆
        if inshape == outshape:
            init_orders = np.eye(inshape // self.feature_dim, outshape // self.feature_dim)
        else:
            init_orders = np.exp(np.random.randn(inshape // self.feature_dim, outshape // self.feature_dim) / 0.01)

        self.inter_orders = tf.Variable(init_orders, dtype=tf.float32)
        self.im = tf.keras.layers.Dense(outshape, use_bias=False, kernel_initializer='normal')

        self.bias_lam = tf.Variable(tf.random.normal((1, self.feature_dim, outshape // self.feature_dim)) * 0.01, dtype=tf.float32)
        self.bias_theta = tf.Variable(tf.random.normal((1, self.feature_dim, outshape // self.feature_dim)) * 0.01, dtype=tf.float32)

        self.drop_ex = tf.keras.layers.Dropout(config.drop_ex)
        self.drop_im = tf.keras.layers.Dropout(config.drop_im)
        self.norm_r = tf.keras.layers.LayerNormalization(axis=-1)
        self.norm_p = tf.keras.layers.LayerNormalization(axis=-1)

    def forward(self, complex_features):
        r, p = complex_features

        lam = r ** 2 + p ** 2 + 1e-8
        theta = tf.atan2(p, r)
        lam, theta = tf.reshape(lam, [lam.shape[0], -1, self.feature_dim]), tf.reshape(theta, [theta.shape[0], -1, self.feature_dim])
        lam = 0.5 * tf.math.log(lam)
        lam, theta = tf.transpose(lam, [0, 2, 1]), tf.transpose(theta, [0, 2, 1])
        lam, theta = self.drop_ex(lam), self.drop_ex(theta)
        lam, theta = tf.matmul(lam, self.inter_orders) + self.bias_lam, tf.matmul(theta, self.inter_orders) + self.bias_theta
        lam = tf.exp(lam)
        lam, theta = tf.transpose(lam, [0, 2, 1]), tf.transpose(theta, [0, 2, 1])

        r, p = tf.reshape(r, [r.shape[0], -1]), tf.reshape(p, [p.shape[0], -1])
        r, p = self.drop_im(r), self.drop_im(p)
        r, p = self.im(r), self.im(p)
        r, p = tf.nn.relu(r), tf.nn.relu(p)
        r, p = tf.reshape(r, [r.shape[0], -1, self.feature_dim]), tf.reshape(p, [p.shape[0], -1, self.feature_dim])

        o_r, o_p = r + lam * tf.cos(theta), p + lam * tf.sin(theta)
        o_r, o_p = tf.reshape(o_r, [o_r.shape[0], -1, self.feature_dim]), tf.reshape(o_p, [o_p.shape[0], -1, self.feature_dim])
        if self.apply_norm:
            o_r, o_p = self.norm_r(o_r), self.norm_p(o_p)
        return o_r, o_p


class EulerNet(object):
    def __init__(self, config):
        self.config = config
        field_num = len(config.feature_stastic) - 1
        shape_list = [config.embedding_size * field_num] + [num_neurons * config.embedding_size for num_neurons in config.order_list]

        interaction_shapes = []
        for inshape, outshape in zip(shape_list[:-1], shape_list[1:]):
            interaction_shapes.append(EulerInteractionLayer(config, inshape, outshape))

        self.Euler_interaction_layers = interaction_shapes
        self.mu = tf.Variable(tf.ones((1, field_num, 1)), dtype=tf.float32)
        self.reg = tf.keras.layers.Dense(1, kernel_initializer='glorot_normal')

    def FeatureInteraction(self, feature, sparse_input, *kargs):
        r, p = self.mu * tf.cos(feature), self.mu * tf.sin(feature)
        for layer in self.Euler_interaction_layers:
            r, p = layer.forward((r, p))
        r, p = tf.reshape(r, [r.shape[0], -1]), tf.reshape(p, [p.shape[0], -1])
        output = self.reg(tf.concat([r, p], axis=1))
        output = tf.sigmoid(output)
        return output