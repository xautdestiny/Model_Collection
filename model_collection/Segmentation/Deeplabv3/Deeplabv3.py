import tensorflow as tf
import keras
from keras.layers import Conv2D, Input, BatchNormalization, MaxPooling2D, ReLU, GlobalAveragePooling2D, Lambda, Concatenate, Reshape, UpSampling2D, Add, AveragePooling2D

def subsample(x, stride):
	if stride == 1:
		return x
	else:
		return MaxPooling2D((1,1), strides=stride)(x)

def Conv2D_same(x, depth, kernel_size, stride, rate=1):
	# if stride == 1:
	# 	return Conv2D(depth, kernel_size, strides=stride, rate=rate, padding='same')(x)
	# else:
	# 	return Conv2D(depth, kernel_size, strides=stride, rate=rate, padding='same')(x)
	return Conv2D(depth, kernel_size, strides=stride, dilation_rate=rate, padding='same', use_bias=False)(x)


def bottleneck(x, depth, depth_bottleneck, stride, rate=1):
	if x.get_shape()[3] == depth:
		shortcut = subsample(x, stride)
	else:
		shortcut = Conv2D(depth, (1,1), strides=stride, use_bias=False)(x)
		shortcut = BatchNormalization()(shortcut)

	residual = Conv2D(depth_bottleneck, (1,1), strides=1, use_bias=False)(x)
	residual = BatchNormalization()(residual)
	residual = ReLU()(residual)

	residual = Conv2D_same(residual, depth_bottleneck, 3, stride, rate=rate)
	residual = BatchNormalization()(residual)
	residual = ReLU()(residual)

	residual = Conv2D(depth, (1,1), strides=1, use_bias=False)(residual)
	residual = BatchNormalization()(residual)

	output = Add()([shortcut, residual])
	output = ReLU()(output)
	return output

def create_deeplabv3(input_shape, num_classes=20):
	## implementation deeplabv3 (ASPP)
	multi_grid = (1,2,4)

	inputs = Input(input_shape)

	x = Conv2D_same(inputs, 64, 7, stride=2)
	x = BatchNormalization()(x)
	x = ReLU()(x)
	x = MaxPooling2D((3,3), strides=2)(x)

	### block 1
	base_depth = 64
	for i in range(2):
		x = bottleneck(x, depth=base_depth*4, depth_bottleneck=base_depth, stride=1)
	x = bottleneck(x, depth=base_depth*4, depth_bottleneck=base_depth, stride=2)


	### block 2
	base_depth = 128
	for i in range(3):
		x = bottleneck(x, depth=base_depth*4, depth_bottleneck=base_depth, stride=1)
	x = bottleneck(x, depth=base_depth*4, depth_bottleneck=base_depth, stride=2)


	### block 3
	base_depth = 256
	for i in range(6): ###23 if resnet101, 36 if resnet152
		x = bottleneck(x, depth=base_depth*4, depth_bottleneck=base_depth, stride=1)

	### block 4
	base_depth = 512
	for i in range(3):
		x = bottleneck(x, depth=base_depth*4, depth_bottleneck=base_depth, stride=1, rate=2*multi_grid[i])

	### ASPP
	aspp = []

	branch_1 = Conv2D(256, (1,1), strides=1, use_bias=False)(x)
	branch_1 = BatchNormalization()(branch_1)
	aspp.append(branch_1)

	for i in range(3):
		branch_2 = Conv2D(256, (3,3), strides=1, dilation_rate=6*(i+1), padding='same', use_bias=False)(x)
		branch_2 = BatchNormalization()(branch_2)
		branch_2 = ReLU()(branch_2)
		aspp.append(branch_2)

	### image level pooling
	# pooled = GlobalAveragePooling2D()(x)
	pooled = AveragePooling2D((32, 32))(x)
	# pooled = Reshape((1,1,base_depth*4))(pooled)
	pooled = Conv2D(256, (1,1), strides=1, use_bias=False)(pooled)
	pooled = BatchNormalization()(pooled)
	pooled = ReLU()(pooled)
	pooled = UpSampling2D((32, 32), interpolation='bilinear')(pooled)
	aspp.append(pooled)

	### fusing
	x = Concatenate()(aspp)
	x = Conv2D(256, (1,1), strides=1, use_bias=False)(x)
	x = BatchNormalization()(x)
	x = ReLU()(x)

	### output logit
	x = Conv2D(num_classes+1, (1,1), strides=1)(x)

	model = keras.models.Model(inputs=inputs, outputs=x)
	return model

if __name__ == "__main__":
	model = create_deeplabv3((512, 512, 3))
	model.summary()
	model.save("deeplabv3_keras.hdf5")