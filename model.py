import torch
import torch.nn as  nn
import torch.nn.functional as F
import ipdb

class Bottleneck(nn.Module):
	expansion = 4
	def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
		super(Bottleneck, self).__init__()
		
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
		self.batch_norm1 = nn.BatchNorm2d(out_channels)
		
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
		self.batch_norm2 = nn.BatchNorm2d(out_channels)
		
		self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
		self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
		
		self.i_downsample = i_downsample
		self.stride = stride
		self.relu = nn.ReLU()
		
	def forward(self, x):
		identity = x.clone()
		x = self.relu(self.batch_norm1(self.conv1(x)))
		
		x = self.relu(self.batch_norm2(self.conv2(x)))
		
		x = self.conv3(x)
		x = self.batch_norm3(x)
		
		#downsample if needed
		if self.i_downsample is not None:
			identity = self.i_downsample(identity)
		#add identity
		x+=identity
		x=self.relu(x)
		
		return x

class Block(nn.Module):
	expansion = 1
	def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
		super(Block, self).__init__()
	   

		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
		self.batch_norm1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
		self.batch_norm2 = nn.BatchNorm2d(out_channels)

		self.i_downsample = i_downsample
		self.stride = stride
		self.relu = nn.ReLU()

	def forward(self, x):
	  identity = x.clone()

	  x = self.relu(self.batch_norm2(self.conv1(x)))
	  x = self.batch_norm2(self.conv2(x))

	  if self.i_downsample is not None:
		  identity = self.i_downsample(identity)
	#   print(x.shape)
	#   print(identity.shape)
	  try:
		  x += identity
	  except:
		  ipdb.set_trace()
	  x = self.relu(x)
	  return x


		
		
class ResNet(nn.Module):
	def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
		super(ResNet, self).__init__()
		in_channels = 8
		self.in_channels = in_channels
		
		self.conv1 = nn.Conv2d(num_channels, in_channels, kernel_size=7, stride=2, padding=3, bias=False)
		self.batch_norm1 = nn.BatchNorm2d(in_channels)
		self.relu = nn.ReLU()
		self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
		
		# self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=in_channels)
		# self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=in_channels*2, stride=2)
		# self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=in_channels*4, stride=2)
		# self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=in_channels*8, stride=2)
		
		# self.avgpool = nn.AdaptiveAvgPool2d((1,1))
		
		# self.fc = nn.Linear(in_channels*2*ResBlock.expansion, num_classes)
		# self.fc = nn.Linear(23104, num_classes)
		N_HIDDEN = 100
		self.fc = torch.nn.Sequential(
			torch.nn.Linear(11552, N_HIDDEN),
			torch.nn.Dropout(0.3),           # drop 50% neurons
			torch.nn.ReLU(),
			torch.nn.Linear(N_HIDDEN, 1)
		)
		
	
	def forward(self, x):
		x = self.relu(self.batch_norm1(self.conv1(x)))
		x = self.max_pool(x)
		
		# x = self.layer1(x)
		# x = self.layer2(x)
		# x = self.layer3(x)
		# x = self.layer4(x)
		# ipdb.set_trace()
		
		# x = self.avgpool(x)
		x = x.reshape(x.shape[0], -1)
		x = self.fc(x)
		
		# return self.activate(x)
		return x
		
	def _make_layer(self, ResBlock, blocks, planes, stride=1):
		ii_downsample = None
		layers = []

		# print(stride, self.in_channels, planes, planes*ResBlock.expansion)
		
		if stride != 1 or self.in_channels != planes*ResBlock.expansion:
			ii_downsample = nn.Sequential(
				nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
				nn.BatchNorm2d(planes*ResBlock.expansion)
			)
			
		layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
		self.in_channels = planes*ResBlock.expansion
		
		# for i in range(blocks-1):
		# 	layers.append(ResBlock(self.in_channels, planes))
		
		# print(len(layers))
		return nn.Sequential(*layers)


def ResNet18(num_classes, channels=3):
	return ResNet(Block, [2,2,2,2], num_classes, channels)

def ResNet50(num_classes, channels=3):
	return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)
	
def ResNet101(num_classes, channels=3):
	return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
	return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)