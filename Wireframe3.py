####what up les bitchs

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt

#deuxieme
import time
import pdb
import importlib
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
import torchvision.models as models
from torch.autograd import Variable

from junc.opts import opts as opts_junc
from junc.utils.plotter import Plotter
import scipy.io as sio
import pickle
import junc.ref as ref

#3eme
from wireframe import Wireframe
from pathlib import Path

###1ere partie

class Residual(nn.Module):
	def __init__(self, numIn, numOut):
		super(Residual, self).__init__()
		self.numIn = numIn
		self.numOut = numOut
		self.bn = nn.BatchNorm2d(self.numIn)
		self.relu = nn.ReLU(inplace = True)
		self.conv1 = nn.Conv2d(self.numIn, self.numOut // 2, bias = True, kernel_size = 1)
		self.bn1 = nn.BatchNorm2d(self.numOut // 2)
		self.conv2 = nn.Conv2d(self.numOut // 2, self.numOut // 2, bias = True, kernel_size = 3, stride = 1, padding = 1)
		self.bn2 = nn.BatchNorm2d(self.numOut // 2)
		self.conv3 = nn.Conv2d(self.numOut // 2, self.numOut, bias = True, kernel_size = 1)

		if self.numIn != self.numOut:
			self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias = True, kernel_size = 1)

	def forward(self, x):
		residual = x
		out = self.bn(x)
		out = self.relu(out)
		out = self.conv1(out)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv3(out)

		if self.numIn != self.numOut:
			residual = self.conv4(x)

		return out + residual

class Hourglass(nn.Module):
    def __init__(self, n, nModules, nFeats):
        super(Hourglass, self).__init__()
        self.n = n
        self.nModules = nModules
        self.nFeats = nFeats

        _up1_, _low1_, _low2_, _low3_ = [], [], [], []
        for j in range(self.nModules):
            _up1_.append(Residual(self.nFeats, self.nFeats))
        self.low1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        for j in range(self.nModules):
            _low1_.append(Residual(self.nFeats, self.nFeats))

        if self.n > 1:
            self.low2 = Hourglass(n - 1, self.nModules, self.nFeats)
        else:
            for j in range(self.nModules):
                _low2_.append(Residual(self.nFeats, self.nFeats))
            self.low2_ = nn.ModuleList(_low2_)

        for j in range(self.nModules):
            _low3_.append(Residual(self.nFeats, self.nFeats))

        self.up1_ = nn.ModuleList(_up1_)
        self.low1_ = nn.ModuleList(_low1_)
        self.low3_ = nn.ModuleList(_low3_)

        self.up2 = nn.Upsample(scale_factor = 2)

    def forward(self, x):
        up1 = x
        for j in range(self.nModules):
            up1 = self.up1_[j](up1)

        low1 = self.low1(x)
        for j in range(self.nModules):
            low1 = self.low1_[j](low1)

        if self.n > 1:
            low2 = self.low2(low1)
        else:
            low2 = low1
            for j in range(self.nModules):
                low2 = self.low2_[j](low2)

        low3 = low2
        for j in range(self.nModules):
            low3 = self.low3_[j](low3)
        up2 = self.up2(low3)

        return up1 + up2

class HourglassNet3D(nn.Module):
    def __init__(self, nStack, nModules, nFeats, nOutChannels):
        super(HourglassNet3D, self).__init__()
        self.nStack = nStack
        self.nModules = nModules
        self.nFeats = nFeats
        self.nOutChannels = nOutChannels
        self.conv1_ = nn.Conv2d(3, 64, bias = True, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.r1 = Residual(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.r4 = Residual(128, 128)
        self.r5 = Residual(128, self.nFeats)

        _hourglass, _Residual, _lin_, _tmpOut, _ll_, _tmpOut_ = [], [], [], [], [], []
        for i in range(self.nStack):
            _hourglass.append(Hourglass(4, self.nModules, self.nFeats))
            for j in range(self.nModules):
                _Residual.append(Residual(self.nFeats, self.nFeats))
            lin = nn.Sequential(nn.Conv2d(self.nFeats, self.nFeats, bias = True, kernel_size = 1, stride = 1),
                                                    nn.BatchNorm2d(self.nFeats), self.relu)
            _lin_.append(lin)
            _tmpOut.append(nn.Conv2d(self.nFeats, self.nOutChannels, bias = True, kernel_size = 1, stride = 1))
            _ll_.append(nn.Conv2d(self.nFeats, self.nFeats, bias = True, kernel_size = 1, stride = 1))
            _tmpOut_.append(nn.Conv2d(self.nOutChannels, self.nFeats, bias = True, kernel_size = 1, stride = 1))

        self.hourglass = nn.ModuleList(_hourglass)
        self.Residual = nn.ModuleList(_Residual)
        self.lin_ = nn.ModuleList(_lin_)
        self.tmpOut = nn.ModuleList(_tmpOut)
        self.ll_ = nn.ModuleList(_ll_)
        self.tmpOut_ = nn.ModuleList(_tmpOut_)

        self.deconv1 = nn.ConvTranspose2d(self.nOutChannels, self.nOutChannels//2, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.nOutChannels//2)
        self.deconv2 = nn.ConvTranspose2d(self.nOutChannels//2, self.nOutChannels//4, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.nOutChannels//4)
        self.conv2 = nn.Conv2d(self.nOutChannels//4, 1, kernel_size=5, stride=1, padding=2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1_(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.r1(x)
        x = self.maxpool(x)
        x = self.r4(x)
        x = self.r5(x)

        out = []

        for i in range(self.nStack):
            hg = self.hourglass[i](x)
            ll = hg
            for j in range(self.nModules):
                ll = self.Residual[i * self.nModules + j](ll)
            ll = self.lin_[i](ll)
            tmpOut = self.tmpOut[i](ll)
            out.append(tmpOut)

            ll_ = self.ll_[i](ll)
            tmpOut_ = self.tmpOut_[i](tmpOut)
            x = x + ll_ + tmpOut_

        shareFeat = out[-1]
        lineOut = self.relu(self.bn2(self.deconv1(shareFeat)))
        lineOut = self.relu(self.bn3(self.deconv2(lineOut)))
        lineOut = self.conv2(lineOut)


        return lineOut

def createModel(opt):
    model = HourglassNet3D(opt.nStack, opt.nModules, opt.nFeats, opt.nOutChannels)

    return model

###2eme partie
__all__ = ['Inception2', 'inception_v2']

def inception_v2(pretrained=False, **kwargs):
    r"""Inception v2 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        model = Inception2(**kwargs)
        #model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
        raise ValueError
        return model

    return Inception2(**kwargs)


class Inception2(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, with_bn = True,  transform_input=False):
        super(Inception2, self).__init__()
        #self.aux_logits = aux_logits
        self.transform_input = transform_input
        # for some task, batch_size is 1 or so, bn might be of no benefit.
        self.with_bn = with_bn
        if with_bn:
            print("Inception_v2 use batch norm")
        else:
            print("Inception_v2 not use batch norm")

        self.Conv2d_1a_7x7 = BasicConv2d(3, 64, with_bn=with_bn, kernel_size=7, stride=2, padding=3)
        self.Conv2d_2b_1x1 = BasicConv2d(64, 64, with_bn=with_bn, kernel_size=1)
        self.Conv2d_2c_3x3 = BasicConv2d(64, 192, with_bn=with_bn,  kernel_size=3, stride=1, padding=1)
        self.Mixed_3b = InceptionD(192, pool_features=32)

        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #        import scipy.stats as stats
        #        stddev = m.stddev if hasattr(m, 'stddev') else 0.1
        #        X = stats.truncnorm(-2, 2, scale=stddev)
        #        values = torch.Tensor(X.rvs(m.weight.data.numel()))
        #        values = values.view(m.weight.data.size())
        #        m.weight.data.copy_(values)
        #    elif isinstance(m, nn.BatchNorm2d):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()

    def forward(self, x):
        x = self.Conv2d_1a_7x7(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.Conv2d_2b_1x1(x)
        x = self.Conv2d_2c_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.Mixed_3b(x)
        
        return x


class InceptionD(nn.Module):

    def __init__(self, in_channels, pool_features, with_bn = False):
        super(InceptionD, self).__init__()
        self.branch0_1x1 = BasicConv2d(in_channels, 64, with_bn = with_bn, kernel_size=1)

        self.branch1_3x3_1 = BasicConv2d(in_channels, 64, with_bn=with_bn, kernel_size=1)
        self.branch1_3x3_2 = BasicConv2d(64, 64, with_bn = with_bn, kernel_size=3, padding=1)

        self.branch2_3x3_1 = BasicConv2d(in_channels, 64, with_bn=with_bn, kernel_size=1)
        self.branch2_3x3_2 = BasicConv2d(64, 96, with_bn=with_bn, kernel_size=3, padding=1)
        self.branch2_3x3_3 = BasicConv2d(96, 96, with_bn=with_bn, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, with_bn=with_bn, kernel_size=1)

    def forward(self, x):
        branch0 = self.branch0_1x1(x)

        branch1 = self.branch1_3x3_1(x)
        branch1 = self.branch1_3x3_2(branch1)

        branch2= self.branch2_3x3_1(x)
        branch2 = self.branch2_3x3_2(branch2)
        branch2 = self.branch2_3x3_3(branch2)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch0, branch1, branch2, branch_pool]
        return torch.cat(outputs, 1)



class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, with_bn=False, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.with_bn = with_bn
        if with_bn:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.00001)

    def forward(self, x):
        x = self.conv(x)
        if self.with_bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)

class inception(nn.Module):
    def __init__(self, classes, opt):
        super(inception, self).__init__()
        self.classes = classes
        self.n_classes = 2
    
        decoder_module = importlib.import_module('junc.model.networks.{}_decoder'.format(opt.decoder))

        self.decoder_ = decoder_module.DecodeNet(opt, 'train')
        self.base_net = inception_v2(num_classes = 2, with_bn=opt.hype.get('batchnorm', True))
        
    def forward(self, im_data, junc_conf, junc_res, bin_conf, bin_res):
        # junc_conf, junc_res, bin_conf, bin_res
        """ input includes:
            im_data
            gt_junctions
            junc_conf
            junc_residual
            bin_conf
            bin_residual
	    """

        batch_size = im_data.size(0)
        base_feat = self.base_net(im_data)
        preds = self.decoder_(base_feat)
        return preds






class BasicConv2d_dec(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d_dec, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x, inplace=True)


class Decoder(nn.Module):
    def __init__(self, input_dim, channel, out1, out2):
        super(Decoder, self).__init__()
        self.nInput = input_dim
        self.conv1 = BasicConv2d_dec(self.nInput, channel, 3, padding=1)
        self.conv1_1 = nn.Conv2d(channel, out1[0], 1, padding=0)
        self.conv1_2 = nn.Conv2d(channel, out1[1], 1, padding=0)

        self.conv2 = BasicConv2d_dec(self.nInput, channel, 3, padding=1)
        self.conv2_1 = nn.Conv2d(channel, out2[0], 1, padding=0)
        self.conv2_2 = nn.Conv2d(channel, out2[1], 1, padding=0)

    def forward(self, input):
        x0 = self.conv1(input)
        junction_logits = self.conv1_1(x0)
        junction_loc = self.conv1_2(x0)

        x1 = self.conv2(input)
        bin_logits = self.conv2_1(x1)
        bin_residual = self.conv2_2(x1)

        return junction_logits, junction_loc, bin_logits, bin_residual


class DecodeNet(nn.Module):
    def __init__(self, opt, phase):
        super(DecodeNet, self).__init__()
        H = opt.hype
        self.batch_size = opt.batch_size
        self.num_bin = H['num_bin']
        self.grid_h = self.grid_w = H['grid_size']
        self.num_grids = self.grid_h * self.grid_w
        self.out_size = self.grid_h * self.grid_w * self.batch_size
        if opt.balance:
            out1 = (3 * H['max_len'], 2 * H['max_len'])
            out2 = (2 * H['num_bin'] * H['max_len'], H['num_bin'] * H['max_len'])
        else:
            out1 = (2 * H['max_len'], 2 * H['max_len'])
            out2 = (2 * H['num_bin'] * H['max_len'], H['num_bin'] * H['max_len'])
        
        decodeFeats = H.get('decodeFeats', 256) # 256 is the reported structure in paper.
        self.decoder = Decoder(decodeFeats, 256, out1, out2)

    def forward(self, input):
        (junction_logits,
         junction_loc,
         bin_logits,
         bin_residual
         ) = self.decoder(input)
        return (
            junction_logits,
            junction_loc,
            bin_logits,
            bin_residual
        )


### WIREFRAME


###
def main():
    with torch.no_grad():
        opt_junc = opts_junc().parse()
        opt_junc.balance = True
        H = opt_junc.hype

        test_junc_thresh_list = [0.1 * x for x in range(10)]
        
        in_ = "IMAGES/1.jpg"
        img = cv2.imread(in_, 1)

        img_line = img/255.
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])
        img_line = np.asarray(img_line)

        img_line[:][:][0] = (img_line[:][:][0] - mean[0].item()) / std[0].item()
        img_line[:][:][1] = (img_line[:][:][1] - mean[1].item()) / std[1].item()
        img_line[:][:][2] = (img_line[:][:][2] - mean[2].item()) / std[2].item()
        img_line = np.transpose(img_line, (2, 0, 1)).astype(np.float32)


        m = torch.load("result/linepx/model_best.pth.tar")
        h = HourglassNet3D(5,2,256,64) #TODO Where is line threshold???
        h.load_state_dict(m)
        del m
        
        img_line = h(torch.from_numpy(img_line).unsqueeze(0))
        img_line = img_line[0].detach().numpy()
        img_line = img_line.transpose((1,2,0))
        
        np.save("zzz_line.npy", img_line)

        """img_line[:][:][0] = (img_line[:][:][0]) * std[0].item() + mean[0].item()
        img_line[:][:][1] = (img_line[:][:][1]) * std[1].item() + mean[1].item()
        img_line[:][:][2] = (img_line[:][:][2]) * std[2].item() + mean[2].item()"""
        plt.imsave("zzz_line.png", img_line[:,:,0])


        img_junc = (img - ref.pixel_mean)
        img_size = img.shape[:2]
        img_junc = cv2.resize(img_junc, (H['image_size'], H['image_size']))
        
        img_junc = img_junc.transpose(2,0,1).astype(np.float32)
        img_junc = torch.from_numpy(img_junc).unsqueeze(0)

        
        junc = inception(['-','+'], opt_junc)
        junc_conf_var = junc_res_var = bin_conf_var = bin_res_var = None
        m = torch.load("output/1/model_17.pth")
        junc.load_state_dict(m)
        del m
        
        (junction_logits,
        junction_loc,
        bin_logits,
        bin_residual
        ) = junc(img_junc, junc_conf_var, junc_res_var, bin_conf_var, bin_res_var)


        def conf_loc(junction_logits, junction_loc, bin_logits, bin_residual, opt):
            junction_logits = junction_logits[:, :2, :, :]
            junc_conf_result = F.softmax(junction_logits, dim=1)
            junc_conf_result = junc_conf_result[:, 1, :, :]
            junc_res_result = junction_loc

            bin_logits = bin_logits.view(-1, 2, opt.hype['num_bin'], opt.hype['grid_size'], opt.hype['grid_size'])
            bin_conf_result = F.softmax(bin_logits, dim=1)
            bin_conf_result = bin_conf_result[:, 1, :, :, :]
            bin_res_result = bin_residual

            return junc_conf_result, junc_res_result, bin_conf_result, bin_res_result

        junc_conf_result, junc_res_result, bin_conf_result, bin_res_result = conf_loc(
            junction_logits, junction_loc, bin_logits, bin_residual, opt_junc)

        plot_ = Plotter(opt_junc.hype)
        tmp2 = {}
        tmp2['h'], tmp2['w'] = img_size
        junc_thresh = test_junc_thresh_list[0] #TODO CHANGE for different tresholds
        (image, tmp2['junctions'], tmp2['thetas'], tmp2['theta_confs']) = plot_.plot_junction_simple(
            img_junc[0].numpy(),
            [junc_conf_result.data.cpu()[0].numpy(),
                junc_res_result.data.cpu()[0].numpy(),
                bin_conf_result.data.cpu()[0].numpy(),
                bin_res_result.data.cpu()[0].numpy()
                ],
            junc_thresh=junc_thresh, 
            theta_thresh=0.5, 
            size_info = img_size,
            keep_conf=True
        )

        cv2.imwrite("zzz.png", image) #imwrite("{}/{}_5.png".format(test_thresh_dir, filename), image)
        sio.savemat("zzz.mat", tmp2) #sio.savemat("{}/{}.mat".format(test_thresh_dir, filename), tmp2)
        with open("zzz.pkl", 'wb') as fn:
            pickle.dump(tmp2, fn)
        




        ### WIREFRAME
        img_dir = Path('IMAGES')
        line_dir = Path("")
        junc_dir = Path("")
        theta_thresh = 0.5
        junc_thresh = 0.5
        exp_name = '1'
        debug = False
        use_mp = False#True if not debug else False
        njobs = 30

        line_thresholds = [2, 6, 10, 20, 30, 50, 80, 100, 150, 200, 250, 255]

        def process_wireframe(exp_name, in_, junc_thresh, line_thresh, debug=False):
            wf = Wireframe(exp_name, in_, junc_thresh=junc_thresh, line_thresh=line_thresh, debug=debug)
            wf.img_dir = img_dir #TODO CHANGE
            wf.line_dir = line_dir #TODO CHANGE
            wf.junc_dir = junc_dir #TODO CHANGE
            wf.load_img("1.jpg") #TODO CHANGE
            wf.load_line("zzz0000") #TODO CHANGE #TODO why remove last 4 characters???
            wf.load_junc("zzz.pkl", theta_thresh=None) #TODO CHANGE
            wf.get_wireframe()
            wf.showLines()

        process_wireframe(exp_name, in_, junc_thresh, line_thresholds[-1]) #TODO Change linetrheshold



if __name__ == "__main__":
    main()

    print("done")