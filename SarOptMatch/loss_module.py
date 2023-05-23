import tensorflow as tf


@tf.function
def crossEntropyNegativeMining(y_true, y_pred):
    """
    Computes the combined loss of the prediction.
    The negative mining term use a soft label.
    The cross entropy use a hard label with only 1 correct matching location.
    bs: batch_size
    """
    bs = tf.shape(y_pred)[0]
    
    # Predicted values inside correct matching region
    matching_region_samples = tf.divide(tf.reduce_sum(tf.multiply(y_pred,y_true),axis=(1,2)),tf.math.count_nonzero(y_true,axis=(1,2),keepdims=False,dtype=tf.float32))

    # Define non-matching region (only look outside correct matching region)
    negative_region = tf.multiply(y_pred,1-y_true)
    negative_region_filterd = tf.reshape(tf.where(tf.equal(negative_region,0.),1.,negative_region),[bs, 65*65])

    # Take n hardest negative samples from non-matching region
    n_neg_samples = 16
    neg_samples =  tf.reduce_mean(-tf.nn.top_k(-negative_region_filterd,k=n_neg_samples)[0],axis=-1) + 1

    # Negative Mining Term
    nm = tf.maximum(-(matching_region_samples-neg_samples),tf.constant(0,dtype=tf.float32))

    # Cross Entropy Term
    xent = tf.nn.softmax_cross_entropy_with_logits(labels = tf.reshape(tf.where(tf.equal(y_true,tf.reduce_max(y_true,axis=(1,2),keepdims=True)),1.,0.),[bs,65*65]),
                                                   logits = tf.reshape(y_pred,[bs,65*65]))

    return xent + nm


@tf.function
def crossEntropy(y_true, y_pred):
    bs = tf.shape(y_pred)[0]
    # Matching can be regarded as a 2-D one-hot classification of 65x65 categories.
    return tf.nn.softmax_cross_entropy_with_logits(labels = tf.reshape(tf.convert_to_tensor(y_true),[bs,65*65]),
                                                   logits = tf.reshape(y_pred,[bs,65*65]))



@tf.function
def windowSum(optical_feature_map, SAR_template_size):
    """
    Returns the integral image of the optical feature map
    where SAR-template_size is the window size of the integral image 
    """
    
    x = tf.cumsum(optical_feature_map,axis=2)
    x = x[:,:,SAR_template_size:-1] -x[:,:,:-SAR_template_size-1]
    x = tf.cumsum(x,axis = 3)
    return (x[:,:,:,SAR_template_size:-1] - x[:,:,:,:-SAR_template_size-1])



@tf.function
def fft_layer(inputs):
    """
    FFT Cross Correlation
    """
    
    opt, sar = inputs
    fft_shape = tf.shape(opt)[2] + tf.shape(sar)[2] - 1
    fft_shape = [fft_shape, fft_shape]
    signal_element_mult = tf.multiply(tf.signal.rfft2d(opt,fft_shape), tf.signal.rfft2d(sar[:,:,::-1,::-1],fft_shape))
    return tf.signal.irfft2d(signal_element_mult,fft_shape)


@tf.function
def Normalization_layer(inputs):
    """
    Normalizes the similarity heatmap.
    Tensorflow implementation of the normalization process in scikit-image.match_template: 
    https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/feature/template.py#L31-L180

    """

    opt,sar,xcorr = inputs

    # Overflow thresholds
    ceil = tf.float32.max
    floor = tf.experimental.numpy.finfo(tf.float32).eps

    # SAR template shape
    float_image_shape = tf.shape(sar)

    # Zero-pad optical floating image to match the dimension of the FFT CC similarity map
    opt = tf.pad(opt,tf.constant([[0,0],[0,0],[192,192],[192,192]]),"CONSTANT")

    sar_volume = tf.cast(tf.math.reduce_prod(float_image_shape[2:]),dtype = tf.float32)
    sar_mean = tf.reduce_mean(sar,axis = [2,3])[:,:,tf.newaxis,tf.newaxis]
    sar_ssd = tf.math.reduce_sum((sar - sar_mean) ** 2, axis = [2,3])[:,:,tf.newaxis,tf.newaxis]

    # Compute integral images
    winsum = windowSum(opt,float_image_shape[2])
    winsum2 = windowSum(opt**2,float_image_shape[2])

    # Normalize
    numerator = tf.subtract(xcorr,winsum*sar_mean)
    winsum2 = tf.subtract(winsum2,tf.multiply(winsum,winsum)/sar_volume)
    winsum2 = tf.experimental.numpy.sqrt(tf.experimental.numpy.maximum(tf.multiply(winsum2,sar_ssd),0))

    # Clip values to avoid overflow
    mask = winsum2 > floor
    return tf.where(mask,tf.divide(tf.clip_by_value(numerator,floor,ceil) , tf.clip_by_value(winsum2,floor,ceil)) ,tf.zeros_like(xcorr))

