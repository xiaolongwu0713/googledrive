import torch
from torch import nn
from torchvision.models import resnet18

'''
Here resNet can be defined with arbitrary input channels instead of only 3 RGB channel.
Also you can specify if load the pretrained model or not. If so, the extra channels should be initialized with some
pretrained channels.
'''
def _my_resnet18(input_channels,num_classes,pretrained=False):
    if pretrained==False:
        model = resnet18(pretrained=False)
        # count_parameters(resnet18)
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc=nn.Linear(in_features=512, out_features=num_classes, bias=True)
        model.add_module("softmax", nn.LogSoftmax(dim=1))
        return model
    else:
        model = resnet18(pretrained=True)
        layer = model.conv1
        # Creating new Conv2d layer
        new_layer = nn.Conv2d(in_channels=input_channels,out_channels=layer.out_channels,kernel_size=layer.kernel_size,
                              stride=layer.stride,padding=layer.padding,bias=layer.bias)

        # initialized the extra channels with some or one pre-trained channel.
        copy_weights = 0

        # Copying the weights from the pretrained to the new layer
        new_layer.weight[:, :layer.in_channels, :, :] = layer.weight.clone()

        # Copying the weights of the `copy_weights` channel of the pretrained layer to the extra channels of the new layer
        for i in range(input_channels - layer.in_channels):
            channel = layer.in_channels + i
            new_layer.weight[:, channel:channel + 1, :, :] = layer.weight[:, copy_weights:copy_weights + 1, :,:].clone()
        new_layer.weight = nn.Parameter(new_layer.weight)

        model.conv1 = new_layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes)
        model.add_module("softmax", nn.LogSoftmax(dim=1))
        return model

# exam model
mm=_my_resnet18(6,5, pretrained=True)

x= torch.randn(1, 6, 224, 224)
mm(x).shape


class my_resnet18(nn.Module):
    #logsoftmax=False + crossenphopy;  logsoftmax=True + NLLLoss
    def __init__(self, input_channels,num_classes,pretrained=False,logsoftmax=False):
        super(my_resnet18, self).__init__()
        self.logsoftmax=logsoftmax
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax(dim=1)

        if pretrained == False:
            self.model = resnet18(pretrained=False)
            if input_channels==3:
                pass
            else:
                # adjust conv1 to the input channels.
                layer = self.model.conv1
                self.model.conv1 = nn.Conv2d(input_channels, out_channels=layer.out_channels, kernel_size=layer.kernel_size
                                             , stride=layer.stride, padding=layer.padding, bias=layer.bias)

            # adjust output class number
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes)
            #self.model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

        else:
            self.model = resnet18(pretrained=True)
            if input_channels==3:
                pass
            else:
                layer = self.model.conv1
                # Creating new Conv2d layer
                new_layer = nn.Conv2d(in_channels=input_channels, out_channels=layer.out_channels,
                                  kernel_size=layer.kernel_size,
                                  stride=layer.stride, padding=layer.padding, bias=layer.bias)

                # initialized the extra channels with some or one pre-trained channel.
                copy_weights = 0

                # Copying the weights from the pretrained to the new layer
                new_layer.weight[:, :layer.in_channels, :, :] = layer.weight.clone()

                # Copying the weights of the `copy_weights` channel of the pretrained layer to the extra channels of the new layer
                for i in range(input_channels - layer.in_channels):
                    channel = layer.in_channels + i
                    new_layer.weight[:, channel:channel + 1, :, :] = layer.weight[:, copy_weights:copy_weights + 1, :,
                                                                     :].clone()
                new_layer.weight = nn.Parameter(new_layer.weight)

                self.model.conv1 = new_layer

            # adjust output class number
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes)
            #self.model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)


    def forward(self, x):
        x = self.model(x)

        if self.logsoftmax:
            if self.training:
                x = self.log_softmax(x)
            else:
                x = self.softmax(x)
        return x


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = backend.int_shape(input)
    residual_shape = backend.int_shape(residual)
    stride_width = int(round(input_shape[2] / residual_shape[2]))
    stride_height = int(round(input_shape[3] / residual_shape[3]))
    equal_channels = input_shape[1] == residual_shape[1]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[1],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                         )(input)

    return add([shortcut, residual])

def DeepConvNet_210519_512_10_res(nb_classes, Chans=64, Samples=256,
                 dropoutRate=0.5):

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(64, (1, 10),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(64, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)


##############################################################################
    block2 = Dropout(dropoutRate)(block1)
    block2 = Conv2D(128, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)

    block2 = _shortcut(block1, block2)
    ##############################################################################

    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block3 = Activation('elu')(block3)

    block3 = Dropout(dropoutRate)(block3)
    block3 = Conv2D(256, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)

    block3 = _shortcut(block2, block3)

    ##############################################################################

    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)
    block4 = Conv2D(512, (1, 10), padding='same', strides=(1, 2),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block4)


    block4 = _shortcut(block3, block4)

    ##############################################################################
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = Dropout(dropoutRate)(block4)

    flatten = GlobalAveragePooling2D(name='avg_pool')(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)




