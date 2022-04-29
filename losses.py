import tensorflow as tf

def mean_gradient_error(targets, outputs, weight=0.1):
    filter_x = tf.tile(
        tf.expand_dims(
            tf.constant([[-1, -2, -2], [0, 0, 0], [1, 2, 1]], dtype = outputs.dtype),
            axis = -1
        ),
        [1, 1, outputs.shape[-1]]
    )

    filter_x = tf.tile(
        tf.expand_dims(filter_x, axis = -1),
         [1, 1, 1, outputs.shape[-1]]
    )

    filter_y = tf.tile(
        tf.expand_dims(
            tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = outputs.dtype),
            axis = -1
        ),
        [1, 1, targets.shape[-1]]
    )

    filter_y = tf.tile(
        tf.expand_dims(filter_y, axis = -1),
        [1, 1, 1, targets.shape[-1]]
    )

    # output gradient
    output_gradient_x = tf.math.square(tf.nn.conv2d(outputs, filter_x, strides = 1, padding = 'SAME'))
    output_gradient_y = tf.math.square(tf.nn.conv2d(outputs, filter_y, strides = 1, padding = 'SAME'))

    #target gradient
    target_gradient_x = tf.math.square(tf.nn.conv2d(targets, filter_x, strides = 1, padding = 'SAME'))
    target_gradient_y = tf.math.square(tf.nn.conv2d(targets, filter_y, strides = 1, padding = 'SAME'))

    # square
    output_gradients = tf.math.sqrt(tf.math.add(output_gradient_x, output_gradient_y))
    target_gradients = tf.math.sqrt(tf.math.add(target_gradient_x, target_gradient_y))

    # compute mean gradient error
    shape = output_gradients.shape[1:3]
    mge = tf.math.reduce_sum(tf.math.squared_difference(output_gradients, target_gradients) / (shape[0] * shape[1]))

    return mge * weight