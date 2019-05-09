import tensorflow as tf
from utils import zero_initializer, normal_initializer


class ConvLayer(object):
    def __init__(self, input_filters, output_filters, act,
                 kernel_size, kernel_stride, kernel_padding):

        super(ConvLayer, self).__init__()

        # number of input channels
        self.input_filters = input_filters

        # number of output channels
        self.output_filters = output_filters

        # convolutional filters kernel size
        self.kernel_size = kernel_size

        # stride of convolutional filters
        self.kernel_stride = kernel_stride

        # padding type of filters
        self.kernel_padding = kernel_padding

        # activation function type
        self.act = act

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):

        #####################################################################
        # TODO: Define Filters and Bias                                     #
        # Filter kernel size is self.kernel_size                            #
        # Number of input channels is self.input_filters                    #
        # Number of desired output filters is self.output_filters           #
        # Define filter tensor with proper size using normal initializer    #
        # Define bias tensor as well using zero initializer                 #
        #####################################################################

        self.conv_filter = normal_initializer(shape=(self.kernel_size, self.kernel_size,
                                                     self.input_filters, self.output_filters),
                                              name='conv_w')
        self.conv_bias = zero_initializer(shape=(self.output_filters,),
                                          name='conv_b')

        #####################################################################
        #                           END OF YOUR CODE                        #
        #####################################################################

        #######################################################################
        # TODO: Apply Convolution, Bias and Activation Function               #
        # Use tf.nn.conv2d and give it following inputs                       #
        #   1. Input tensor                                                   #
        #   2. Filter you have defined in above empty part                    #
        #   3. Stride tensor showing stride size for each dimension           #
        #   4. Padding type based on self.kernel_padding                      #
        # Add bias after filtering by convolutions                            #
        # Finally apply activation function and store it as self.total_output #
        #######################################################################

        self.conv_output = tf.nn.conv2d(input=inputs,
                                        filter=self.conv_filter,
                                        strides=[1, self.kernel_stride, self.kernel_stride, 1],
                                        padding=self.kernel_padding)

        self.total_output = self.act(self.conv_output + self.conv_bias)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return self._call(self.total_output)


class ConvPoolLayer(ConvLayer):
    def __init__(self, input_filters, output_filters, act,
                 kernel_size, kernel_stride, kernel_padding,
                 pool_size, pool_stride, pool_padding):

        # Calling ConvLayer constructor will store convolutional section config
        super(ConvPoolLayer, self).__init__(input_filters, output_filters, act,
                                            kernel_size, kernel_stride, kernel_padding)

        # size of kernel in pooling
        self.pool_size = pool_size

        # size of stride in pooling
        self.pool_stride = pool_stride

        # type of padding in pooling
        self.pool_padding = pool_padding

    def _call(self, inputs):

        ##########################################################################
        # TODO: Apply Pooling                                                    #
        # Please note that when __call__ method is called for an object of this  #
        # class, convolution operation will be applied on original input.        #
        # We override _call function so that the result convolution will later   #
        # move through max pooling function which should be defined below.       #
        # To do so, use tf.nn.max_pool and give it following inputs:             #
        #   1. Input tensor                                                      #
        #   2. Kernel size for max pooling                                       #
        #   3. Stride tensor showing stride size for each dimension              #
        #   4. Padding type based on self.kernel_padding                         #
        # Please store output in self.pooling_output                             #
        ##########################################################################
        self.pooling_output = tf.nn.max_pool(value=inputs,
                                             ksize=[1, self.pool_size, self.pool_size, 1],
                                             strides=[1, self.pool_stride, self.pool_stride, 1],
                                             padding=self.pool_padding)
        ##########################################################################
        #                           END OF YOUR CODE                             #
        ##########################################################################
        return self.pooling_output


class DeconvLayer(object):
    def __init__(self, input_filters, output_filters, act,
                 kernel_size, kernel_stride, kernel_padding):

        super(DeconvLayer, self).__init__()

        # number of input channels
        self.input_filters = input_filters

        # number of output channels
        self.output_filters = output_filters

        # transposed convolutional filters kernel size
        self.kernel_size = kernel_size

        # stride of transposed convolutional filters
        self.kernel_stride = kernel_stride

        # padding type of filters
        self.kernel_padding = kernel_padding

        # activation function type
        self.act = act

    def __call__(self, inputs):

        ############################################################################################
        # TODO: Define Filters and Bias                                                            #
        # Filter kernel size is self.kernel_size                                                   #
        # Number of input channels is self.input_filters                                           #
        # Number of desired output filters is self.output_filters                                  #
        # Define filter tensor with proper size using normal initializer                           #
        # Note that tensor shape of this filter is different from that of the filter in ConvLayer  #
        # Define bias tensor as well using zero initializer                                        #
        ############################################################################################

        self.deconv_filter = normal_initializer(shape=(self.kernel_size, self.kernel_size,
                                                       self.output_filters, self.input_filters),
                                                name='conv_w')
        self.deconv_bias = zero_initializer(shape=(self.output_filters,),
                                            name='conv_b')

        ############################################################################################
        #                           END OF YOUR CODE                                               #
        ############################################################################################

        # input height and width
        input_height = inputs.get_shape().as_list()[1]
        input_width = inputs.get_shape().as_list()[2]

        ############################################################################
        # TODO: Calculate Output Shape                                             #
        # Use input height and width to set output height and width respectively   #
        # The formula to calculate output shapes depends on type of padding        #
        ############################################################################

        if self.kernel_padding == 'SAME':
            output_height = input_height * self.kernel_stride
            output_width = input_width * self.kernel_stride
        elif self.kernel_padding == 'VALID':
            output_height = (input_height - 1) * self.kernel_stride + self.kernel_size
            output_width = (input_width - 1) * self.kernel_stride + self.kernel_size
        else:
            raise Exception('No such padding')

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        #########################################################################
        # TODO: Apply Transposed Convolution, Bias and Activation Function      #
        # Use tf.nn.conv2d_transpose and give it following inputs               #
        #   1. Input tensor                                                     #
        #   2. Filter you have defined above                                    #
        #   3. Output shape you have calculated above                           #
        #   4. Stride tensor showing stride size for each dimension             #
        #   5. Padding type based on self.kernel_padding                        #
        # Add bias after filtering by transposed convolutions                   #
        # Finally apply activation function and store it as self.total_output   #
        #########################################################################

        self.deconv_output = tf.nn.conv2d_transpose(value=inputs,
                                                    filter=self.deconv_filter,
                                                    output_shape=tf.stack([tf.shape(inputs)[0],
                                                                           output_height, output_width,
                                                                           self.output_filters]),
                                                    strides=[1, self.kernel_stride, self.kernel_stride, 1],
                                                    padding=self.kernel_padding)

        self.total_output = self.act(self.deconv_output + self.deconv_bias)

        #########################################################################
        #                             END OF YOUR CODE                          #
        #########################################################################

        return self.total_output
