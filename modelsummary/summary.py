import math


__all__ = [
    'Module', 'Identity', 'Linear', 'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d',
    'ConvTranspose2d', 'ConvTranspose3d', 'Threshold', 'ReLU', 'Hardtanh', 'ReLU6',
    'Sigmoid', 'Tanh', 'Softmax', 'Softmax2d', 'LogSoftmax', 'ELU', 'SELU', 'CELU', 'GLU', 'GELU', 'Hardshrink',
    'LeakyReLU', 'LogSigmoid', 'Softplus', 'Softshrink', 'MultiheadAttention', 'PReLU', 'Softsign', 'Softmin',
    'Tanhshrink', 'RReLU', 'AvgPool1d', 'AvgPool2d', 'AvgPool3d', 'MaxPool1d', 'MaxPool2d',
    'MaxPool3d', 'MaxUnpool1d', 'MaxUnpool2d', 'MaxUnpool3d', 'FractionalMaxPool2d', "FractionalMaxPool3d",
    'LPPool1d', 'LPPool2d', 'LocalResponseNorm', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'InstanceNorm1d',
    'InstanceNorm2d', 'InstanceNorm3d', 'LayerNorm', 'GroupNorm', 'SyncBatchNorm',
    'Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d', 'AlphaDropout', 'FeatureAlphaDropout',
    'ReflectionPad1d', 'ReflectionPad2d', 'ReflectionPad3d', 'ReplicationPad2d', 'ReplicationPad1d', 'ReplicationPad3d',
    'CrossMapLRN2d', 'Embedding', 'EmbeddingBag', 'RNNBase', 'RNN', 'LSTM', 'GRU', 'RNNCellBase', 'RNNCell',
    'LSTMCell', 'GRUCell', 'PixelShuffle', 'PixelUnshuffle', 'Upsample', 'UpsamplingNearest2d', 'UpsamplingBilinear2d',
    'PairwiseDistance', 'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d', 'AdaptiveAvgPool1d',
    'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d', 'TripletMarginLoss', 'ZeroPad2d', 'ConstantPad1d', 'ConstantPad2d',
    'ConstantPad3d', 'Bilinear', 'CosineSimilarity', 'Unfold', 'Fold',
    'AdaptiveLogSoftmaxWithLoss', 'TransformerEncoder', 'TransformerDecoder',
    'TransformerEncoderLayer', 'TransformerDecoderLayer', 'Transformer',
    'LazyLinear', 'LazyConv1d', 'LazyConv2d', 'LazyConv3d',
    'LazyConvTranspose1d', 'LazyConvTranspose2d', 'LazyConvTranspose3d',
    'LazyBatchNorm1d', 'LazyBatchNorm2d', 'LazyBatchNorm3d',
    'LazyInstanceNorm1d', 'LazyInstanceNorm2d', 'LazyInstanceNorm3d',
    'Flatten', 'Unflatten', 'Hardsigmoid', 'Hardswish', 'SiLU', 'Mish', 'TripletMarginWithDistanceLoss', 'ChannelShuffle'
]


def parameter_summary(model,border=True):
    '''
    Args: 
        model: PyTorch model
        border: Seperation line after printing out 
        the details of each layer, default = False
    Returns: 
        summary of all the number of parameters of the model
    '''
    total_params = 0
    s="{:<20}     {:^20} {}  {:>20}\n".format('LAYER TYPE','KERNEL SHAPE', '#parameters',' (weights+bias)')
    s += "_"*100 + "\n"
    
    for index,i in enumerate(model.modules()):
        if i._get_name() in __all__:
            if border: s += "_"*100 + "\n"
            layer=i._get_name()
            if bool(i._parameters.keys()):
                weight_shape = list(i._parameters['weight'].shape)
                x = math.prod(weight_shape)
       
                if i._parameters['bias'] is None:
                    i._parameters['bias']=[]
                
                bias = len(i._parameters['bias'])
                total_params+= x+bias
                
                s += " {:<20}   {:^20}\t{:,}  {:>25}\n".format(layer+'-'+str(index),str(weight_shape),x+bias,f'({x} + {bias})')
                # if i._parameters['bias'] is not None:
                #     bias = list(i._parameters['bias'].shape)
                #     print(x+bias[0])
                # else:
                #     total_params+= x
                #     print(x)
    
            else:    
                s += " {:<20}   {:^20}\t{:}  {:>25}\n".format(layer+'-'+str(index),'-','-','-')

    s += "="*100 +"\n"
    print(s)           
    print('Total parameters {:,}'.format(total_params))
    return total_params
    
