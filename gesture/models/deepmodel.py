import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import elu
from torch.nn import functional as F

from braindecode.models.modules import Expression, AvgPool2dWithConv, Ensure4d
from braindecode.models.functions import identity, transpose_time_to_spat, squeeze_final_output
from braindecode.util import np_to_var
from braindecode.models import ShallowFBCSPNet,EEGNetv4,Deep4Net

from common_dl import add_channel_dimm,init_weights
from gesture.models.utils import squeeze_all

class expand_dim(torch.nn.Module):
    def forward(self, x):
        while(len(x.shape) < 4):
            x = x.unsqueeze(1)
        return x

def swap_time_spat(x):
    """Swap time and spatial dimensions.

    Returns
    -------
    x: torch.Tensor
        tensor in which last and first dimensions are swapped
    """
    return x.permute(0, 1, 3, 2)

#model = deepnet(n_chans,n_classes,input_window_samples=input_window_samples,final_conv_length='auto',)
# expect input shape: (batch_size, channel_number, time_length)
class deepnet(nn.Sequential):
    """
    Deep ConvNet model from [1]_.

    References
    ----------

    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(self,in_chans,n_classes,input_window_samples,final_conv_length,n_filters_time=64,n_filters_spat=64,filter_time_length=50,
        pool_time_length=3,pool_time_stride=3,n_filters_2=50,filter_length_2=10,n_filters_3=50,filter_length_3=10,n_filters_4=50,
        filter_length_4=10,first_nonlin=elu,first_pool_mode="max",first_pool_nonlin=identity,later_nonlin=elu,later_pool_mode="max",
        later_pool_nonlin=identity,drop_prob=0.5,double_time_convs=False,split_first_layer=True,batch_norm=True,batch_norm_alpha=0.1,
        stride_before_pool=False,
    ):
        super().__init__()
        if final_conv_length == "auto":
            assert input_window_samples is not None
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.final_conv_length = final_conv_length
        self.n_filters_time = n_filters_time
        self.n_filters_spat = n_filters_spat
        self.filter_time_length = filter_time_length
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.n_filters_2 = n_filters_2
        self.filter_length_2 = filter_length_2
        self.n_filters_3 = n_filters_3
        self.filter_length_3 = filter_length_3
        self.n_filters_4 = n_filters_4
        self.filter_length_4 = filter_length_4
        self.first_nonlin = first_nonlin
        self.first_pool_mode = first_pool_mode
        self.first_pool_nonlin = first_pool_nonlin
        self.later_nonlin = later_nonlin
        self.later_pool_mode = later_pool_mode
        self.later_pool_nonlin = later_pool_nonlin
        self.drop_prob = drop_prob
        self.double_time_convs = double_time_convs
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.stride_before_pool = stride_before_pool

        if self.stride_before_pool:
            conv_stride = self.pool_time_stride
            pool_stride = 1
        else:
            conv_stride = 1
            pool_stride = self.pool_time_stride
        self.add_module("Add_dim", expand_dim())
        pool_class_dict = dict(max=nn.MaxPool2d, mean=AvgPool2dWithConv)
        first_pool_class = pool_class_dict[self.first_pool_mode]
        later_pool_class = pool_class_dict[self.later_pool_mode]
        if self.split_first_layer:
            self.add_module("dimshuffle", Expression(swap_time_spat))
            self.add_module("conv_time",nn.Conv2d(1,self.n_filters_time,(self.filter_time_length, 1),stride=1,),)
            self.add_module("conv_spat",nn.Conv2d(self.n_filters_time,self.n_filters_spat,(1, self.in_chans),
                                                  stride=(conv_stride, 1),bias=not self.batch_norm,),)
            n_filters_conv = self.n_filters_spat
        else:
            self.add_module("conv_time",nn.Conv2d(self.in_chans,self.n_filters_time,(self.filter_time_length, 1),
                                                  stride=(conv_stride, 1),bias=not self.batch_norm,),)
            n_filters_conv = self.n_filters_time
        if self.batch_norm:
            self.add_module("bnorm",nn.BatchNorm2d(n_filters_conv,momentum=self.batch_norm_alpha,affine=True,eps=1e-5,),)
        self.add_module("conv_nonlin", Expression(self.first_nonlin)) #elu
        self.add_module("pool",first_pool_class(kernel_size=(self.pool_time_length, 1), stride=(pool_stride, 1)),) #MaxPool2d
        #self.add_module("pool_nonlin", Expression(self.first_pool_nonlin)) # identity

        def add_conv_pool_block(model, n_filters_before, n_filters, filter_length, block_nr):
            suffix = "_{:d}".format(block_nr)
            self.add_module("drop" + suffix, nn.Dropout(p=self.drop_prob))
            self.add_module("conv" + suffix,nn.Conv2d(n_filters_before,n_filters,(filter_length, 1),
                    stride=(conv_stride, 1),bias=not self.batch_norm,),)
            if self.batch_norm:
                self.add_module("bnorm" + suffix,nn.BatchNorm2d(n_filters,momentum=self.batch_norm_alpha,affine=True,eps=1e-5,),)
            self.add_module("nonlin" + suffix, Expression(self.later_nonlin)) # elu

            # maxpool2d
            #self.add_module("pool" + suffix,later_pool_class(kernel_size=(self.pool_time_length, 1),stride=(pool_stride, 1),),)

            #Expression(expression=identity)
            #self.add_module("pool_nonlin" + suffix, Expression(self.later_pool_nonlin)) # identity

        add_conv_pool_block(self, n_filters_conv, self.n_filters_2, self.filter_length_2, 2)
        add_conv_pool_block(self, self.n_filters_2, self.n_filters_3, self.filter_length_3, 3)
        add_conv_pool_block(self, self.n_filters_3, self.n_filters_4, self.filter_length_4, 4)

        self.add_module("last_drop", nn.Dropout(p=self.drop_prob))

        # self.add_module('drop_classifier', nn.Dropout(p=self.drop_prob))
        self.eval()
        if self.final_conv_length == "auto":
            out = self(np_to_var(np.ones((1, self.in_chans, self.input_window_samples),dtype=np.float32,)))
            n_channels=out.cpu().data.numpy().shape[1]
            n_out_time,n_out_spatial = out.cpu().data.numpy().shape[2],out.cpu().data.numpy().shape[3]
            self.final_conv_length = n_out_time

        #self.add_module("conv_classifier",nn.Conv2d(self.n_filters_4,self.n_classes,(self.final_conv_length, 1),bias=True,),)
        self.add_module("globalAvgPooling",nn.AvgPool2d((n_out_time,n_out_spatial)))
        self.add_module("squeeze1", Expression(squeeze_all))
        self.add_module("conv_classifier", nn.Linear(n_channels,n_classes))
        self.add_module("softmax", nn.LogSoftmax(dim=1))
        #self.add_module("squeeze2", Expression(squeeze_final_output))

        # Initialization, xavier is same as in our paper...
        # was default from lasagne
        init.xavier_uniform_(self.conv_time.weight, gain=1)
        # maybe no bias in case of no split layer and batch norm
        if self.split_first_layer or (not self.batch_norm):
            init.constant_(self.conv_time.bias, 0)
        if self.split_first_layer:
            init.xavier_uniform_(self.conv_spat.weight, gain=1)
            if not self.batch_norm:
                init.constant_(self.conv_spat.bias, 0)
        if self.batch_norm:
            init.constant_(self.bnorm.weight, 1)
            init.constant_(self.bnorm.bias, 0)
        param_dict = dict(list(self.named_parameters()))
        for block_nr in range(2, 5):
            conv_weight = param_dict["conv_{:d}.weight".format(block_nr)]
            init.xavier_uniform_(conv_weight, gain=1)
            if not self.batch_norm:
                conv_bias = param_dict["conv_{:d}.bias".format(block_nr)]
                init.constant_(conv_bias, 0)
            else:
                bnorm_weight = param_dict["bnorm_{:d}.weight".format(block_nr)]
                bnorm_bias = param_dict["bnorm_{:d}.bias".format(block_nr)]
                init.constant_(bnorm_weight, 1)
                init.constant_(bnorm_bias, 0)

        init.xavier_uniform_(self.conv_classifier.weight, gain=1)
        init.constant_(self.conv_classifier.bias, 0)

        # Start in eval mode
        self.eval()


def shortcut(block):  # @save
    channels = [(i.in_channels, i.out_channels) for i in block if type(i) == nn.Conv2d]
    input_channels = channels[0][0]
    num_channels = channels[0][1]
    strides = [i.stride for i in block if type(i) == nn.Conv2d]
    stride = strides[0][0] * strides[0][1]
    # 1,pad when using non-1 sized kernel; 2, make sure the shortcut stride = conv stride
    if (not input_channels == num_channels) or (not stride == 1):  # when channel number or width/height changed
        _shortcut = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=stride)
    else:
        _shortcut = Expression(expression=identity)
    return _shortcut

# model = deepnet(n_chans,n_classes,input_window_samples=input_window_samples,final_conv_length='auto',)
class deepnet_resnet(nn.Module):
    def __init__(self,in_chans,n_classes,input_window_samples,n_filters_time=64,n_filters_spat=64,expand=True,
        filter_time_length=50,drop_prob=0.5,pool_time_length=3,pool_time_stride=3,n_filters_2=50,filter_length_2=10,n_filters_3=50,
        filter_length_3=10,n_filters_4=50,filter_length_4=10,first_nonlin=elu,first_pool_mode="max",first_pool_nonlin=identity,
        later_nonlin=elu,later_pool_mode="max",later_pool_nonlin=identity,double_time_convs=False,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.n_filters_time = n_filters_time
        self.n_filters_spat = n_filters_spat
        self.filter_time_length = filter_time_length
        self.drop_prob = drop_prob
        if expand==True:
            conv_channels = [64,128,256,512,1024]
        else:
            conv_channels = [64, 50,50,50,50]
            #conv_channels = [64, 64, 64, 64, 64]

        self.block0 = nn.Sequential(add_channel_dimm(),
                                    nn.Conv2d(1,self.n_filters_time,(1,self.filter_time_length),stride=1,),
                                    nn.Conv2d(self.n_filters_time,self.n_filters_spat,(self.in_chans,1),stride=(1, 1)),
                                    nn.BatchNorm2d(self.n_filters_spat,affine=True,eps=1e-5,),
                                    nn.ELU(),
                                    nn.MaxPool2d(kernel_size=(1,3), stride=(1,3)))

        kernel=11
        padding=11//2
        stride=2
        self.block1 = nn.Sequential(nn.Dropout(p=self.drop_prob),
                           nn.Conv2d(conv_channels[0],conv_channels[1], (1, kernel),padding=(0,padding), stride=(1,stride)))

        self.block2=nn.Sequential(nn.BatchNorm2d(conv_channels[1], affine=True, eps=1e-5, ),
                             nn.ELU(),
                             nn.Dropout(p=self.drop_prob),
                             nn.Conv2d(conv_channels[1],conv_channels[2], (1,kernel),padding=(0,padding),stride=(1,stride))
                             )

        self.block3 = nn.Sequential(nn.BatchNorm2d(conv_channels[2], affine=True, eps=1e-5, ),
                               nn.ELU(),
                               nn.Dropout(p=self.drop_prob),
                               nn.Conv2d(conv_channels[2],conv_channels[3], (1, kernel),padding=(0,padding), stride=(1, stride))
                               )

        self.block4 = nn.Sequential(nn.BatchNorm2d(conv_channels[3], affine=True, eps=1e-5, ),
                               nn.ELU(),
                               nn.Dropout(p=self.drop_prob),
                               nn.Conv2d(conv_channels[3],conv_channels[4], (1, kernel),padding=(0,padding), stride=(1, stride))
                               )

        self.sc1 = shortcut(self.block1)
        self.sc2 = shortcut(self.block2)
        self.sc3 = shortcut(self.block3)
        self.sc4 = shortcut(self.block4)

        x = torch.randn(1, in_chans, input_window_samples)
        all=nn.Sequential(self.block0,self.block1,self.block2,self.block3,self.block4)
        out=all(x)
        out_channels,n_out_time,n_out_spatial = out.shape[1],out.shape[2], out.shape[3]


        #self.add_module("squeeze2", Expression(squeeze_final_output))
        self.block_final = nn.Sequential(nn.AvgPool2d((n_out_time,n_out_spatial)),
                                         Expression(squeeze_all),
                                         nn.Linear(out_channels,n_classes))
        self.softmax = nn.LogSoftmax()

        self.apply(init_weights)

        # Start in eval mode
        self.eval()



    def forward(self, x):
        x = self.block0(x) #torch.Size([32, 64, 1, 150])

        y = self.block1(x) #torch.Size([32, 50, 11, 75])
        ff= self.sc1(x)
        x=y+ff

        y = self.block2(x)
        ff= self.sc2(x)
        x = y + ff

        y = self.block3(x)
        ff= self.sc3(x)
        x = y + ff

        y = self.block4(x)
        ff= self.sc4(x)
        x = y + ff

        x = self.block_final(x)

        return self.softmax(x)



