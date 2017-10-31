import mxnet as mx


def conv(net,
         channels,
         filter_dimension,
         stride,
         weight=None,
         bias=None,
         act_type="relu",
         no_bias=False,
         name=None
         ):
    # 2d convolution's input should have the shape of 4D (batch_size,1,seq_len,feat_dim)
    net = mx.sym.Convolution(data=net, num_filter=channels, kernel=filter_dimension, stride=stride, weight=weight,
                             bias=bias, no_bias=no_bias, name=name)
    net = mx.sym.Activation(data=net, act_type=act_type)
    return net
