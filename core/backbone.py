#! /usr/bin/env python
# coding=utf-8


import core.common as common
import tensorflow as tf

def _stem_block_1(input_data, is_training=True,num_init_channel=32):
    
    conv0 =  common.convolutional(input_data, (3, 3, 3,  num_init_channel), is_training, 'stem_block_conv0',downsample=True)

    conv1_l0 = common.convolutional(conv0, (1,1,num_init_channel, int(num_init_channel / 2)), is_training,'stem_block_conv1_l0')

    conv1_l1 = common.convolutional(conv1_l0, (3,3,int(num_init_channel / 2),num_init_channel), is_training,'stem_block_conv1_l1',downsample=True)

    maxpool1_r0 = tf.nn.max_pool(conv0, (1,2,2,1), (1,2,2,1), padding='SAME', name='stem_block_maxpool1_r0')

    filter_concat = tf.concat([conv1_l1, maxpool1_r0], axis=-1)

    output = common.convolutional(filter_concat, (1,1,2*num_init_channel,num_init_channel), is_training, 'stem_block_output')


    return output

def _dense_block_1(input_x, stage, num_block, k, bottleneck_width, is_training=True, reuse=False):

    output = input_x

    for index in range(num_block):
        dense_block_name = 'stage_{}_dense_block_{}'.format(stage, index)
        with tf.variable_scope(dense_block_name) as scope:
            if reuse:
                scope.reuse_variables()


            inter_channel = k * bottleneck_width

            conv_left_0 = common.convolutional(output, (1,1,output.get_shape().as_list()[-1], inter_channel) , is_training, 'stem_block_conv0_1')

            conv_left_1 = common.convolutional(conv_left_0, (3, 3, inter_channel,k),is_training, 'conv_left_1')
            conv_left_1 = common.cbam_block(conv_left_1, "cbam_left")

            conv_right_0 = common.convolutional(output,  (1, 1, output.get_shape().as_list()[-1], inter_channel), is_training, 'conv_right_0')

            conv_right_1 = common.convolutional(conv_right_0, (3, 3, inter_channel, k), is_training, 'conv_right_1')

            conv_right_2 = common.convolutional(conv_right_1, (3, 3, k, k), is_training, 'conv_right_2')



            conv_right_2 = common.cbam_block(conv_right_2, "cbam_right")

            output = tf.concat([output, conv_left_1, conv_right_2], axis=3)

    return output


def _transition_layer_1(input_x, stage, output_channel, is_avgpool=True, is_training=True,reuse=False):

    transition_layer_name = 'stage_{}_transition_layer'.format(stage)

    with tf.variable_scope(transition_layer_name) as scope:
        if reuse:
            scope.reuse_variables()
        conv0 = common.convolutional(input_x, (1,1,input_x.get_shape().as_list()[-1], output_channel),is_training,'transition_layer_conv0')
        if is_avgpool:
            is_training = tf.cast(True, tf.bool)
            output = common.separable_conv('transition_layer_avgpool', conv0, output_channel, output_channel,
                                           is_training, downsample=True)
            # output = common.convolutional(conv0, filters_shape=(3, 3, output_channel, output_channel),
            #                                   trainable=is_training, name='transition_layer_avgpool', downsample=True)
            # output = slim.avg_pool2d(conv0, 2, 2, scope='transition_layer_avgpool')
        else:
            output = conv0


        output = common.cbam_block(output, "output_0")
    return output









def peleetnet_yolov3(input_x,trainable):


    stem_block_output= _stem_block_1(input_x, trainable , num_init_channel=32)

    # stem_block_output = common.convolutional(stem_block_output, filters_shape=(3, 3, 32, 64), trainable=trainable, name='conv1')
    #
    # stem_block_output = common.convolutional(stem_block_output, filters_shape=(1, 1, 64, 32), trainable=trainable, name='conv2')

    dense_block_output = _dense_block_1(stem_block_output, 0, 3, 16, 2)

    transition_layer_output = _transition_layer_1(dense_block_output, 0, 128)

    dense_block_output1 = _dense_block_1(transition_layer_output, 1, 4, 16, 2)

    transition_layer_output1 = _transition_layer_1(dense_block_output1, 1, 256)

    dense_block_output2 = _dense_block_1(transition_layer_output1, 2, 8, 16, 4)

    transition_layer_output2 = _transition_layer_1(dense_block_output2, 2, 512)

    dense_block_output3 = _dense_block_1(transition_layer_output2, 3, 16, 16, 4)

    transition_layer_output3 = _transition_layer_1(dense_block_output3, 3, 1024, is_avgpool=False)

    return dense_block_output1, dense_block_output2, transition_layer_output3


def darknet53(input_data, trainable):

    with tf.variable_scope('darknet'):

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  3,  32), trainable=trainable, name='conv0')
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32,  64),
                                          trainable=trainable, name='conv1', downsample=True)

        for i in range(1):
            input_data = common.residual_block(input_data,  64,  32, 64, trainable=trainable, name='residual%d' %(i+0))

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  64, 128),
                                          trainable=trainable, name='conv4', downsample=True)

        for i in range(2):
            input_data = common.residual_block(input_data, 128,  64, 128, trainable=trainable, name='residual%d' %(i+1))

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),
                                          trainable=trainable, name='conv9', downsample=True)

        for i in range(8):
            input_data = common.residual_block(input_data, 256, 128, 256, trainable=trainable, name='residual%d' %(i+3))

        route_1 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512),
                                          trainable=trainable, name='conv26', downsample=True)

        for i in range(8):
            input_data = common.residual_block(input_data, 512, 256, 512, trainable=trainable, name='residual%d' %(i+11))

        route_2 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024),
                                          trainable=trainable, name='conv43', downsample=True)

        for i in range(4):
            input_data = common.residual_block(input_data, 1024, 512, 1024, trainable=trainable, name='residual%d' %(i+19))

        return route_1, route_2, input_data




