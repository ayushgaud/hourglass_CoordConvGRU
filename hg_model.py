import tensorflow as tf
from convGRU_cell import ConvGRUCell
class HourGlass():

    def __init__(self, is_training, batch_norm_decay,
                 batch_norm_epsilon,
                 data_format='channels_first',
                 num_stacks=2, 
                 num_out=256, 
                 n_low=4,
                 num_joints=36,
                 batch_size=1):

        self.data_format = data_format
        self.num_out = num_out
        self.num_stacks = num_stacks
        self.n_low = n_low
        self.num_joints = num_joints
        self.is_training = is_training
        self.batch_norm_epsilon = batch_norm_epsilon
        self.batch_size = batch_size
        self.curr_n_low_id = 0
        self.curr_stack_id = 0

        if data_format == 'channels_first':
            self.bn_axis = 1
        else:
            self.bn_axis = -1

    def _conv_block(self, inputs, num_out):
        
        with tf.name_scope('conv_block') as name_scope:
            norm1 = tf.layers.batch_normalization(inputs=inputs, training=self.is_training, axis=self.bn_axis, epsilon=self.batch_norm_epsilon)
            conv1 = tf.layers.conv2d(inputs=norm1,filters=num_out/2,kernel_size=(1,1), data_format=self.data_format, activation=tf.nn.relu)
            norm2 = tf.layers.batch_normalization(inputs=conv1, training=self.is_training, axis=self.bn_axis, epsilon=self.batch_norm_epsilon)
            if self.data_format == 'channels_first':
                pad = tf.pad(norm2, tf.constant([[0,0],[0,0],[1,1],[1,1]]))
            else:
                pad = tf.pad(norm2, tf.constant([[0,0],[1,1],[1,1],[0,0]]))
            conv2 = tf.layers.conv2d(inputs=pad,filters=num_out/2,kernel_size=(3,3), data_format=self.data_format, activation=tf.nn.relu)
            norm3 = tf.layers.batch_normalization(inputs=conv2, training=self.is_training, axis=self.bn_axis, epsilon=self.batch_norm_epsilon)
            conv3 = tf.layers.conv2d(inputs=norm3,filters=num_out,kernel_size=(1,1), data_format=self.data_format, activation=tf.nn.relu)
            # tf.logging.info('image after unit %s: %s', name_scope, conv3.get_shape())
        return conv3

    def _skip_layer(self, inputs, num_out):

        if inputs.get_shape().as_list()[3] == num_out:
            return inputs
        else:
            return tf.layers.conv2d(inputs=inputs,filters=num_out,kernel_size=(1,1), data_format=self.data_format)

    def _recurrent_cell(self, n_low, num_out, name):

        cell = ConvGRUCell(shape=[4*2**n_low,4*2**n_low],
                           filters=num_out,
                           kernel=[3, 3],
                           activation=tf.nn.relu,
                           data_format=self.data_format,
                           is_training=self.is_training,
                           name=name)

        return cell

    def _convgru_layer(self, inputs, input_state, cell):

        with tf.name_scope('convgru') as name_scope:
            # print('n_low %d n_stack %d' %(self.curr_n_low_id, self.curr_stack_id))
            # print(inputs,input_state)
            if input_state == None:
                output, output_state = cell(inputs, cell.zero_state(self.batch_size, dtype=inputs.dtype))
            else:
                output, output_state = cell(inputs, input_state)
            tf.logging.info('image after unit %s: %s', name_scope, output.get_shape())
            # tf.logging.info('state after unit %s: %s', name_scope, output_state.get_shape())
            return output, output_state

    def _residual(self, inputs, num_out, is_recurrent=False):
        with tf.name_scope('residual') as name_scope:
            convb = self._conv_block(inputs, num_out)
            if self.curr_n_low_id > 0 and is_recurrent:
                cell = self.cells[self.curr_n_low_id - 1][self.curr_stack_id]
                input_state = self.input_states[self.curr_n_low_id - 1][self.curr_stack_id]
                output, output_state = self._convgru_layer(inputs, input_state, cell)
                self.output_states[self.curr_n_low_id - 1][self.curr_stack_id] = output_state
                output = tf.add_n([convb, output])
            else:
                skipl = self._skip_layer(inputs, num_out)
                output = tf.add_n([convb, skipl])
            tf.logging.info('image after unit %s: %s', name_scope, output.get_shape())
        return output

    def _hourglass(self, inputs, num_out, n_low):

        with tf.name_scope('hg') as name_scope:
            self.curr_n_low_id = n_low
            up_1 = self._residual(inputs, num_out, is_recurrent=True)
            pool_1 = tf.layers.max_pooling2d(inputs=inputs, pool_size=(2,2), strides=(2,2), data_format=self.data_format)
            low_1 = self._residual(pool_1, num_out)

            if n_low > 1:
                low_2 = self._hourglass(low_1, num_out, n_low - 1)
            else:
                low_2 = self._residual(low_1, num_out)

            low_3 = self._residual(low_2, num_out)

            upsampling = tf.keras.layers.UpSampling2D(data_format=self.data_format)
            up_2 = upsampling.apply(inputs=low_3)

            output = tf.add_n([up_1, up_2])
        tf.logging.info('image after unit %s: %s', name_scope, output.get_shape())
        return output

    def _linear_layer(self, inputs, num_out):
        
        conv = tf.layers.conv2d(inputs=inputs,filters=num_out,kernel_size=(1,1), data_format=self.data_format)
        norm = tf.layers.batch_normalization(inputs=conv, training=self.is_training, axis=self.bn_axis, epsilon=self.batch_norm_epsilon)
        return tf.nn.relu(norm)

    def _occ_layer(self, inputs, num_out):
        with tf.name_scope('occ') as name_scope:
            conv = self._conv_block(inputs, num_out)
            assert inputs.get_shape().ndims == 4
            if self.data_format == 'channels_first':
              pool = tf.reduce_mean(conv, [2, 3])
            else:
              pool = tf.reduce_mean(conv, [1, 2])
            output = tf.layers.dense(pool, self.num_joints)
            tf.logging.info('image after unit %s: %s', name_scope, output.get_shape())
        return output

    def model(self, inputs, cells, input_states):

        self.cells = cells
        self.input_states = input_states
        self.output_states = [[None]*self.num_stacks for _ in range(self.n_low)]
        
        num_out = self.num_out
        outputs = [None] * self.num_stacks
        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs,[0,3,1,2])

        conv_1 = self._linear_layer(inputs, num_out/4)
        res_1 = self._residual(conv_1, num_out/2)
        res_2 = self._residual(res_1, num_out/2)
        inter = self._residual(res_2, num_out)
        
        # Generate stacked hourglass 
        for i in range(self.num_stacks):
            tf.logging.info('HourGlass stack %d', i)
            self.curr_stack_id = i
            hg = self._hourglass(inter, num_out, self.n_low)
            lin_layer = self._linear_layer(hg, num_out)

            # Predicted heatmaps
            hm = tf.layers.conv2d(inputs=lin_layer,filters=self.num_joints,kernel_size=(1,1), data_format=self.data_format)
            outputs[i] = hm

            # Add predictions back
            if i < self.num_stacks:
                lin_layer_ = tf.layers.conv2d(inputs=lin_layer,filters=num_out,kernel_size=(1,1), data_format=self.data_format)
                hm_ = tf.layers.conv2d(inputs=hm,filters=num_out,kernel_size=(1,1), data_format=self.data_format)
                inter = tf.add_n([inter, lin_layer_, hm_])
        if self.data_format == 'channels_first':
            model = tf.transpose(tf.stack(outputs),[1,0,3,4,2])
        else:
            model = tf.transpose(tf.stack(outputs),[1,0,2,3,4])
        return model, self.output_states
