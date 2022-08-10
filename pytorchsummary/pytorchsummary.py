import torch
from torch import nn
from collections import Counter

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

def summary(input_size,model,_print=True,border=False)->dict:
    '''
    Args: 
        input_size: tuple (#channels,height,width) input 
            image/tensor dimension WITHOUT batch_size
        model: PyTorch model
        border: Seperation line after printing out 
            the details of each layer, default = True
        _print: default==True , if set to False
            it won't print the summary, it will just 
            return the number of parameters (values)
    Returns: 
        A tuple
        (Total-TRAINABLE-params, total-params, total-NON-trainable-params)
    '''

    image = torch.rand((1,)+input_size)

    if next(model.parameters()).is_cuda:
        image = image.to('cuda')

    h=[]
    activation ={}
    global total_params
    global non_trainable
    total_params = 0
    non_trainable = 0

    def getShape(module):
        
        def hook(module,input,output):

            global total_params
            global non_trainable

            if module._get_name() in __all__:

                name = module._get_name() + f'-{len(activation)+1}'
                activation[name] = [str(list(output.shape))]

                    
                if bool(module._parameters.keys()):
                    weight_shape= torch.tensor(module._parameters['weight'].shape)
                    _weight_shape = list(module._parameters['weight'].shape)
                    x = torch.prod(weight_shape).item()
                    Wgrad = module._parameters['weight'].requires_grad
                    if not Wgrad: non_trainable+=x

                    if module._parameters['bias'] is not None:
                        bias = list(module._parameters['bias'].shape)
                        total_params += x+bias[0]
                        Bgrad = module._parameters['bias'].requires_grad
                        if not Bgrad: non_trainable+=bias[0]
                        
                        activation[name].append(str(_weight_shape))
                        activation[name].append(str(x+bias[0]))
                        activation[name].append(f'({x} + {bias[0]})')
                        activation[name].append(f'{Wgrad} {Bgrad}')

                    else:

                        activation[name].append(str(_weight_shape))
                        activation[name].append(str(x))
                        activation[name].append(f'({x}+0)')
                        activation[name].append(str(Wgrad))
                        total_params+= x
                else:   

                    activation[name].append('')
                    activation[name].append('')
                    activation[name].append('')
                    activation[name].append('')
    
        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)):
            h.append(module.register_forward_hook(hook))   
        


    model.apply(getShape)
    _ = model(image)

    for i in h:i.remove()
    
    heading = ['Layer','Output Shape','Kernal Shape','#params','#(weights + bias)','requires_grad']
    s='{:>20}\t{:<20}\t{:^20}\t{:20}\t{:20}\t{:^10}\n'.format(*heading)
    s+='-'*150+'\n'

    for k,v in activation.items():
        s+='{:>20}\t{:<20}\t{:^20}\t{:20}\t{:20}\t{:^10}\n'.format(k,*v)
        if border:s+='_'*150 +'\n'
    
    s+='_'*150 +'\n'
    if _print:
        print(s)           
        print('Total parameters {:,}'.format(total_params))
        print('Total Non-Trainable parameters {:,}'.format(non_trainable))
        print('Total Trainable parameters {:,}'.format(total_params-non_trainable))
    
    return (total_params-non_trainable,total_params,non_trainable)

def get_num_layers(model)->dict:
    l =[]
    for i in model.modules():
        name=i._get_name()
        if name in __all__:
            l.append(name)
    return dict(Counter(l))