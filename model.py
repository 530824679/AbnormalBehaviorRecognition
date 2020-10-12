import numpy as np
from ops import *

"""
So, the total number of layers is (3*blokcs)*residual_layer_num + 2
because, blocks = split(conv 2) + transition(conv 1) = 3 layer
and, first conv layer 1, last dense layer 1
thus, total number of layers = (3*blocks)*residual_layer_num + 2
"""
class Network():
    def __init__(self, num_classes, is_train, reduction_ratio, block, cardinality, depth):
        self.is_train = is_train
        self.num_classes = num_classes
        self.reduction_ratio = reduction_ratio
        self.block = block                      # res block (split + transition)
        self.cardinality = cardinality          # split number
        self.depth = depth                      # out channel

    def first_layer(self, input, scope):
        with tf.name_scope(scope):
            net = ConvLayer(input, filter=64, kernel=[3, 3], stride=1, scope=scope+'_conv1')
            net = BatchNormalization(net, is_train=self.is_train, scope=scope+'_batch1')
            net = Relu(net, scope=scope+'_relu1')
            return net

    def squeeze_excitation_layer(self, input, out_dim, ratio, scope):
        """
        Squeeze Excitation
        arguments:
            input(tf.Tensor): a 4D tensor
              4D tensor with shape:
              `(batch_size, channels, rows, cols)`
            out_dim: input channels num
            ratio: full connect reduction ratio
            :return: tf.Tensor: input of global infomation. input*excitation
        """
        with tf.name_scope(scope):
            squeeze = GlobalAveragePool(input)

            excitation = Fully_connected(squeeze, num_classes=out_dim / ratio, scope=scope+'_fully_connected1')
            excitation = Relu(excitation, scope=scope+'_relu')
            excitation = Fully_connected(excitation, num_classes=out_dim, scope=scope+'_fully_connected2')
            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            scale = input * excitation

            return scale

    def transform_layer(self, input, stride, scope):
        with tf.name_scope(scope):
            net = ConvLayer(input, filter=self.depth, kernel=[1, 1], stride=1, scope=scope + '_conv1')
            net = BatchNormalization(net, is_train=self.is_train, scope=scope + '_batch1')
            net = Relu(net, scope=scope + '_relu1')

            net = ConvLayer(net, filter=self.depth, kernel=[3, 3], stride=stride, scope=scope + '_conv2')
            net = BatchNormalization(net, is_train=self.is_train, scope=scope + '_batch2')
            net = Relu(net, scope=scope + '_relu2')
            return net

    def split_layer(self, input, stride, scope):
        with tf.name_scope(scope):
            layers_split = list()
            for i in range(self.cardinality):
                splits = self.transform_layer(input, stride=stride, scope=scope + '_splitN_' + str(i))
                layers_split.append(splits)
            return Concat(layers_split)

    def transition_layer(self, input, out_dim, scope):
        with tf.name_scope(scope):
            net = ConvLayer(input, filter=out_dim, kernel=[1, 1], stride=1, scope=scope+'_conv1')
            net = BatchNormalization(net, is_train=self.is_train, scope=scope+'_batch1')
            return net

    def residual_layer(self, input, out_dim, layer_num, res_block=3):
        # split + transform + transition + merge

        for i in range(res_block):
            input_dim = int(np.shape(input)[-1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1

            net = self.split_layer(input, stride=stride, scope='split_layer_'+layer_num+'_'+str(i))
            net = self.transition_layer(net, out_dim=out_dim, scope='trans_layer_'+layer_num+'_'+str(i))
            net = self.squeeze_excitation_layer(net, out_dim=out_dim, ratio=self.reduction_ratio, scope='squeeze_layer_'+layer_num+'_'+str(i))

            if flag is True:
                pad_input_x = AveragePool(input)
                pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]])
            else:
                pad_input_x = input

            output = Relu(net + pad_input_x, scope='residual_relu_layer_'+layer_num)
        return output

    def build_network(self, input):
        net = self.first_layer(input, scope='first_layer')

        net = self.residual_layer(net, out_dim=64, layer_num='res_1', res_block=self.block)
        net = self.residual_layer(net, out_dim=128, layer_num='res_2', res_block=self.block)
        net = self.residual_layer(net, out_dim=256, layer_num='res_3', res_block=self.block)

        net = GlobalAveragePool(net)
        net = flatten(net)

        net = Fully_connected(net, num_classes=self.num_classes, scope='fully_connected')
        return net
