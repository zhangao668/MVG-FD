# -*- coding: utf-8 -*-
# @Author  : Lintao Peng
# @File    : Ushape_Trans.py
# coding=utf-8
# Design based on the pix2pix

import torch.nn as nn
import torch.nn.functional as F 
import torch
import datetime
import os
import time
import timeit
import copy
import numpy as np 
from torch.nn import ModuleList
from torch.nn import Conv2d
from torch.nn import LeakyReLU

from net.FD import DWT, UpSample, ARB, ContrastEnhancedHighFrequency
from net.block import *
from net.block import _equalized_conv2d
from net.SGFMT import TransformerModel
from net.PositionalEncoding import FixedPositionalEncoding,LearnedPositionalEncoding
from net.CMSFFT import ChannelTransformer
from net.MVG import *








def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)






class Generator(nn.Module):

	def __init__(self,
		# 	图片维度
		img_dim=256,
		patch_dim=16,
		embedding_dim=512,
		num_channels=3,
		# 改
		deep_channel=3,
		in_deep_ch=3,

		num_heads=8,
		num_layers=4,
		hidden_dim=256,
		dropout_rate=0.0,
		attn_dropout_rate=0.0,
		in_ch=3, 
		out_ch=3,
		conv_patch_representation=True,
		positional_encoding_type="learned",
		use_eql=True):
		super(Generator, self).__init__()

		assert embedding_dim % num_heads == 0
		assert img_dim % patch_dim == 0

		self.deep_channel =deep_channel
		self.in_deep_ch=in_deep_ch

		self.out_ch=out_ch
		self.in_ch=in_ch
		self.img_dim = img_dim
		self.embedding_dim = embedding_dim
		self.num_heads = num_heads
		self.patch_dim = patch_dim
		self.num_channels = num_channels
		self.dropout_rate = dropout_rate
		self.attn_dropout_rate = attn_dropout_rate
		self.conv_patch_representation = conv_patch_representation  #True


		self.num_patches = int((img_dim // patch_dim) ** 2)
		self.seq_length = self.num_patches
		self.flatten_dim = 128 * num_channels

        #线性编码
		self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
		#位置编码
		if positional_encoding_type == "learned":
			self.position_encoding = LearnedPositionalEncoding(
				self.seq_length, self.embedding_dim, self.seq_length
			)
		elif positional_encoding_type == "fixed":
			self.position_encoding = FixedPositionalEncoding(
				self.embedding_dim,
			)

		self.pe_dropout = nn.Dropout(p=self.dropout_rate)

		self.transformer = TransformerModel(
			embedding_dim, #512
			num_layers, #4
			num_heads,  #8

			hidden_dim,  #4096

			self.dropout_rate,
			self.attn_dropout_rate,
        )

		#layer Norm
		self.pre_head_ln = nn.LayerNorm(embedding_dim)

		if self.conv_patch_representation:

			self.Conv_x = nn.Conv2d(
				256,
				self.embedding_dim,  #512
				kernel_size=3,
				stride=1,
				padding=1
		    )

		self.bn = nn.BatchNorm2d(256)
		self.relu = nn.ReLU(inplace=True)



		#modulelist

		self. rgb_to_feature=ModuleList([from_rgb(32),from_rgb(64),from_rgb(128)])

		self.feature_to_rgb=ModuleList([to_rgb(32),to_rgb(64),to_rgb(128),to_rgb(256)])

		self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

		self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.rgbd_to_feature = ModuleList([from_rgbd(32), from_rgbd(64), from_rgbd(128)])


		# self.Maxpoold = nn.MaxPool2d(kernel_size=2, stride=2)

		self.Maxpoold1= nn.MaxPool2d(kernel_size=2, stride=2)
		self.Maxpoold2= nn.MaxPool2d(kernel_size=2, stride=2)
		self.ContrastEnhancedHighFrequency=ContrastEnhancedHighFrequency()

		self.DWT=DWT()
		self.UpsSample = UpSample()
		self.ARB=ARB(32,False)
		self.one_conv_e1_hh = nn.Conv2d(64, 32, kernel_size=1)

		self.ResidualWithLearnedAlpha=ResidualWithLearnedAlpha()



		self.Convd1=conv_block(self.in_deep_ch,16)
		self.Convd1_1=conv_block(16,32)
		self.Prompt_block1=Prompt_block(32,64)


		self.Convd2 = conv_block(32, 32)
		self.Convd2_1 = conv_block(32, 64)
		self.Prompt_block2 = Prompt_block(64, 128)


		self.Convd3 = conv_block(64,64)
		self.Convd3_1 = conv_block(64,128)
		self.Prompt_block3 = Prompt_block(128, 256)

		self.Convd1_1_1=conv_block(64,64)
		self.Convd2_2_2 = conv_block(128, 128)
		self.Convd3_3_3 = conv_block(256, 256)


		self.Conv1=conv_block(self.in_ch, 16)
		self.Conv1_1 = conv_block(16, 32)


		self.Conv2 = conv_block(32, 32)
		self.Conv2_1 = conv_block(32, 64)

		self.Conv3 = conv_block(64,64)
		self.Conv3_1 = conv_block(64,128)

		self.Conv4 = conv_block(128,128)
		self.Conv4_1 = conv_block(128,256)


		self.Conv5 = conv_block(512,256)


		self.mtc = ChannelTransformer(channel_num=[32,64,128,256],
									patchSize=[32, 16, 8, 4])

		self.Up5 = up_conv(256, 256)

		self.coatt5 = CCA(F_g=256, F_x=256)

		self.Up_conv5 = conv_block(512, 256)
		self.Up_conv5_1 = conv_block(256, 256)

		self.Up4 = up_conv(256, 128)

		self.coatt4 = CCA(F_g=128, F_x=128)
		self.Up_conv4 = conv_block(256, 128)
		self.Up_conv4_1 = conv_block(128, 128)

		self.Up3 = up_conv(128, 64)
		self.coatt3 = CCA(F_g=64, F_x=64)
		self.Up_conv3 = conv_block(128, 64)
		self.Up_conv3_1 = conv_block(64, 64)

		self.Up2 = up_conv(64, 32)
		self.coatt2 = CCA(F_g=32, F_x=32)
		self.Up_conv2 = conv_block(64, 32)
		self.Up_conv2_1 = conv_block(32, 32)

		self.Conv = nn.Conv2d(32, self.out_ch, kernel_size=1, stride=1, padding=0)

		# self.active = torch.nn.Sigmoid()
		# 
	def reshape_output(self,x): #将transformer的输出resize为原来的特征图尺寸
		x = x.view(
			x.size(0),
			int(self.img_dim / self.patch_dim),
			int(self.img_dim / self.patch_dim),
			self.embedding_dim,
			)#B,16,16,512
		x = x.permute(0, 3, 1, 2).contiguous()

		return x

	def forward(self, x, d):


		output=[]

		# 池化操作
		x_1=self.Maxpool(x)

		x_2=self.Maxpool(x_1)

		x_3=self.Maxpool(x_2)


		dpx_1=self.Maxpool(d)

		dpx_2=self.Maxpool(dpx_1)




		e1 = self.Conv1(x)

		e1 = self.Conv1_1(e1)

		f1=e1
		wf1=self.DWT(f1)

		wf1=self.ContrastEnhancedHighFrequency(wf1[1],wf1[2],wf1[3])

		HH_up=self.ARB(wf1)
		HH_up =self.UpsSample(HH_up)

		e1 = self.ResidualWithLearnedAlpha(e1,HH_up)




		dp1=self.Convd1(d)
		dp1=self.Convd1_1(dp1)

		e1_dp1 = torch.cat([e1,dp1],dim=1)
		e1_dp1=self.Convd1_1_1(e1_dp1)

		e1_prompt = self.Prompt_block1(e1_dp1)

		e1=e1+e1_prompt




		e2 = self.Maxpool1(e1)

		x_1=self.rgb_to_feature[0](x_1)

		e2=x_1+e2


		e2 = self.Conv2(e2)
		e2 = self.Conv2_1(e2)



		dp2 = self.Maxpoold1(dp1)
		dpx_1=self.rgbd_to_feature[0](dpx_1)
		dp2 = dp2+dpx_1

		dp2=self.Convd2(dp2)
		dp2=self.Convd2_1(dp2)

		e2_dp2 = torch.cat([e2, dp2], dim=1)

		e2_dp2 = self.Convd2_2_2(e2_dp2)
		e2_prompt = self.Prompt_block2(e2_dp2)
		e2 = e2+e2_prompt





		e3 = self.Maxpool2(e2)

		x_2=self.rgb_to_feature[1](x_2)

		e3=x_2+e3

		e3 = self.Conv3(e3)

		e3 = self.Conv3_1(e3)



		dp3=self.Maxpoold2(dp2)

		dpx_2=self.rgbd_to_feature[1](dpx_2)

		dp3=dp3+dpx_2
		dp3=self.Convd3(dp3)
		dp3=self.Convd3_1(dp3)

		e3_dp3 = torch.cat([e3, dp3], dim=1)

		e3_dp3=self.Convd3_3_3(e3_dp3)

		e3_prompt = self.Prompt_block3(e3_dp3)
		e3=e3+e3_prompt







		e4 = self.Maxpool3(e3)

		x_3=self.rgb_to_feature[2](x_3)

		e4=x_3+e4
		e4 = self.Conv4(e4)
		e4 = self.Conv4_1(e4)

		e5 = self.Maxpool4(e4)



		e1,e2,e3,e4,att_weights = self.mtc(e1,e2,e3,e4)



		#spatial-wise transformer-based attention
		residual=e5

		e5= self.bn(e5)
		e5=self.relu(e5)
		e5= self.Conv_x(e5) #out->512*16*16 shape->B,512,16,16
		e5= e5.permute(0, 2, 3, 1).contiguous()  # B,512,16,16->B,16,16,512

		e5= e5.view(e5.size(0), -1, self.embedding_dim) #B,16,16,512->B,16*16,512
		e5= self.position_encoding(e5)
		e5= self.pe_dropout(e5)
		# apply transformer
		e5= self.transformer(e5)
		e5= self.pre_head_ln(e5)	
		e5= self.reshape_output(e5)#out->512*16*16 shape->B,512,16,16
		e5=self.Conv5(e5) #out->256,16,16 shape->B,256,16,16

		e5=e5+residual



		d5 = self.Up5(e5)

		e4_att = self.coatt5(g=d5, x=e4)

		d5 = torch.cat((e4_att, d5), dim=1)



		d5 = self.Up_conv5(d5)
		d5 = self.Up_conv5_1(d5)


		out3=self.feature_to_rgb[3](d5)
		output.append(out3)#32*32orH/8,W/8


		d4 = self.Up4(d5)

		e3_att = self.coatt4(g=d4, x=e3)
		d4 = torch.cat((e3_att, d4), dim=1)

		d4 = self.Up_conv4(d4)
		d4 = self.Up_conv4_1(d4)

		out2=self.feature_to_rgb[2](d4)

		output.append(out2)#64*64orH/4,W/4

		d3 = self.Up3(d4)

		e2_att = self.coatt3(g=d3, x=e2)
		d3 = torch.cat((e2_att, d3), dim=1)

		d3 = self.Up_conv3(d3)
		d3 = self.Up_conv3_1(d3)

		out1=self.feature_to_rgb[1](d3)

		output.append(out1)#128#128orH/2,W/2

		d2 = self.Up2(d3)

		e1_att = self.coatt2(g=d2, x=e1)
		d2 = torch.cat((e1_att, d2), dim=1)
		d2 = self.Up_conv2(d2)
		d2 = self.Up_conv2_1(d2)
		#32
		out0=self.feature_to_rgb[0](d2)
		output.append(out0)#256*256


		return output



# 鉴别器
class Discriminator(nn.Module):
    def __init__(self, in_channels=3,use_eql=True):
        super(Discriminator, self).__init__()

        self.use_eql=use_eql
        self.in_channels=in_channels


        #modulelist

        self.rgb_to_feature1=ModuleList([from_rgb(32),from_rgb(64),from_rgb(128)])
        self.rgb_to_feature2=ModuleList([from_rgb(32),from_rgb(64),from_rgb(128)])


        self.layer=_equalized_conv2d(self.in_channels*2, 64, (1, 1), bias=True)

        self.pixNorm = PixelwiseNorm()
        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)
        self.layer0=DisGeneralConvBlock(64,64,use_eql=self.use_eql)
        self.layer1=DisGeneralConvBlock(128,128,use_eql=self.use_eql)
        self.layer2=DisGeneralConvBlock(256,256,use_eql=self.use_eql)
        self.layer3=DisGeneralConvBlock(512,512,use_eql=self.use_eql)
        self.layer4=DisFinalBlock(512,use_eql=self.use_eql)

        


    def forward(self, img_A, inputs):



        x=torch.cat((img_A[3], inputs[3]), 1)
        y = self.pixNorm(self.lrelu(self.layer(x)))

        y=self.layer0(y)

        

        x1=self.rgb_to_feature1[0](img_A[2])
        x2=self.rgb_to_feature2[0](inputs[2])
        x=torch.cat((x1,x2),1)
        y=torch.cat((x,y),1)
        y=self.layer1(y)


        x1=self.rgb_to_feature1[1](img_A[1])
        x2=self.rgb_to_feature2[1](inputs[1])
        x=torch.cat((x1,x2),1)
        y=torch.cat((x,y),1)
        y=self.layer2(y)



        x1=self.rgb_to_feature1[2](img_A[0])
        x2=self.rgb_to_feature2[2](inputs[0])
        x=torch.cat((x1,x2),1)
        y=torch.cat((x,y),1)
        y=self.layer3(y)
        #16*16*512
        y=self.layer4(y)


        return y





