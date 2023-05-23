import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Dropout, concatenate, add, multiply, Input,Conv2DTranspose, BatchNormalization, Lambda, Activation, UpSampling2D, Permute, Cropping2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import os

from . import loss_module
from . import utils

#%% Blocks for regular U-Net

def EncoderBlock(inputs, n_filters = 32):

    conv = Conv2D(n_filters, 3, activation='relu',padding='same')(inputs)
    conv = BatchNormalization()(conv)

    conv = Conv2D(n_filters, 3, activation='relu',padding='same')(conv)
    conv = BatchNormalization()(conv)

    next_layer = MaxPooling2D(pool_size = (2,2))(conv)       
    return next_layer, conv


def DecoderBlock(prev_layer_input, skip_layer_input, n_filters = 32):
    
    up = Conv2DTranspose(n_filters, (2,2), strides=(2,2),activation='relu',padding='same')(prev_layer_input)
    up = BatchNormalization()(up)
    merge = Concatenate(axis=3)([up, skip_layer_input])

    conv = Conv2D(n_filters, 3, activation ='relu',padding='same')(merge)
    conv = BatchNormalization()(conv)

    conv = Conv2D(n_filters, 3 , activation ='relu',padding='same')(conv)
    conv = BatchNormalization()(conv)
    
    return conv



#%% Blocks for U-Net with Residual convs and Attention gates

def ResidualBlock(inputs, n_filters = 32, channel_axis = 3, activation = 'relu'):
    """
    Residual block with 2x Residual conv -> BatchNorm -> Act
    """

    conv = Conv2D(n_filters, 3, padding='same',kernel_initializer='HeNormal')(inputs)
    conv = BatchNormalization(axis = channel_axis)(conv)
    conv = Activation(activation)(conv)

    conv = Conv2D(n_filters, 3, padding='same',kernel_initializer='HeNormal')(conv)
    conv = BatchNormalization(axis = channel_axis)(conv)
    conv = Activation(activation)(conv)

    shortcut = Conv2D(n_filters, 1, padding='same')(inputs)  
    residual_path = add([shortcut, conv])
    out = BatchNormalization(axis = channel_axis)(residual_path)
    out = Activation(activation)(out)
    return Dropout(0.1)(out) 

def GatingBlock(inputs, output_dim, channel_axis = 3, activation = 'relu'):
    """
    Resizes input channel dimensions to output_dim
    """

    conv = Conv2D(output_dim, 1, padding='same', kernel_initializer='HeNormal')(inputs)    
    conv = BatchNormalization(axis = channel_axis)(conv)
    return Activation(activation)(conv)


def repeat_channel(input, repeat_count):
    """
    repeat input feature channel repeat_count times
    """

    return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': repeat_count})(input)


def AttentionBlock(input, gate_in, out_shape, activation = 'relu'):
    """
    Attention mechanism.
    Modified version of code from https://github.com/MoleImg/Attention_UNet/blob/master/AttResUNet.py
    """

    # Useful resources:
    # https://github.com/nabsabraham/focal-tversky-unet/blob/master/newmodels.py
    # https://medium.com/aiguys/attention-u-net-resunet-many-more-65709b90ac8b
    
       
    shape_x = K.int_shape(input)

    theta_x = Conv2D(out_shape, (2, 2), strides=(2, 2), padding='same',kernel_initializer='HeNormal')(input) 

    phi_g = Conv2D(out_shape, (1, 1), padding='same',kernel_initializer='HeNormal')(gate_in)
    
    # The two vectors are summed element-wise. 
    # This process results in aligned weights becoming larger while unaligned weights becoming relatively smaller.
    concat_xg = add([phi_g, theta_x])
       
    act_xg = Activation(activation)(concat_xg)
    
    # The resultant vector goes through a ReLU activation layer and a 1x1 convolution that collapses the dimensions to 1
    psi = Conv2D(1, (1, 1), padding='same',kernel_initializer='HeNormal')(act_xg)
    
    # This vector goes through a sigmoid layer which scales the vector between the range [0,1],
    # producing the attention coefficients (weights), where coefficients closer to 1 indicate more relevant features.
    sigmoid_xg = Activation('sigmoid')(psi)
    
    # The attention coefficients are upsampled to the original dimensions of the x vector
    upsample_psi = UpSampling2D(size=(2,2))(sigmoid_xg)
    upsample_psi = repeat_channel(upsample_psi, shape_x[3])

    y = multiply([upsample_psi, input])

    # According to https://towardsdatascience.com/a-detailed-explanation-of-the-attention-u-net-b371a5590831
    # a final 1x1x1 convolution is used to consolidate the attention signal to original x dimensions
    result = Conv2D(shape_x[3], (1, 1), padding='same', kernel_initializer='HeNormal')(y)
    result_bn = BatchNormalization()(result)
    return result_bn 


def ResidualDecoderBlock(gate_layer, attention_layer, n_filters = 32, channel_axis = 3, attention = True, activation = "relu"):
    """
    Applies attention and upsamples gate-layer before applying residual block
    """
    
    if attention:
        gate = GatingBlock(gate_layer, n_filters, activation = activation)
        attention = AttentionBlock(attention_layer, gate, n_filters)
        up = UpSampling2D(size = (2,2), interpolation="bilinear")(gate_layer)
        up = concatenate([up,attention], axis=channel_axis)
        up_conv = ResidualBlock(up, n_filters, activation = activation)
        return up_conv
    
    else:        
        # Without attention mechanism
        up = UpSampling2D(size = (2,2), interpolation="bilinear")(gate_layer)
        up = concatenate([up,attention_layer], axis=channel_axis)
        up_conv = ResidualBlock(up, n_filters, activation = activation)
        return up_conv




# %% Architectures


def ResUNet_Attention(input_shape, n_filters = 32,  attention = True, activation = "relu"):
    """
    U-Net with Residual convolutions and Attention Gates
    """
    ins = Input(input_shape)
    # Down
    c1 = ResidualBlock(ins, n_filters, activation = activation)
    c1_pool = MaxPooling2D(pool_size = (2,2))(c1)
    c2 = ResidualBlock(c1_pool, n_filters*2, activation = activation)
    c2_pool = MaxPooling2D(pool_size = (2,2))(c2)
    c3 = ResidualBlock(c2_pool, n_filters*4, activation = activation)
    c3_pool = MaxPooling2D(pool_size = (2,2))(c3)
    c4 = ResidualBlock(c3_pool, n_filters*8, activation = activation)
    c4_pool = MaxPooling2D(pool_size = (2,2))(c4)

    #Bottleneck
    c5 = ResidualBlock(c4_pool, n_filters*16, activation = activation)

    # Up
    u4 = ResidualDecoderBlock(c5,c4,n_filters*8, attention = attention, activation = activation)
    u3 = ResidualDecoderBlock(u4,c3,n_filters*4, attention = attention, activation = activation)
    u2 = ResidualDecoderBlock(u3,c2,n_filters*2, attention = attention, activation = activation)
    u1 = ResidualDecoderBlock(u2,c1,n_filters, attention = attention, activation = activation)
    
    return Model(inputs=[ins], outputs=[u1])


def ResUNet_Attention_toy_example(input_shape, n_filters = 16):
    """
    U-Net with Residual convolutions and Attention Gates
    """
    ins = Input(input_shape)
    # Down
    c1 = ResidualBlock(ins,n_filters)
    c1_pool = MaxPooling2D(pool_size = (2,2))(c1)

    #Bottleneck
    c5 = ResidualBlock(c1_pool, n_filters*2)

    # Up
    u1 = ResidualDecoderBlock(c5, c1, n_filters)
    
    return Model(inputs=[ins], outputs=[u1])



def unet(input_shape, n_filters= 32):
    """
    Implementation of the U-Net described in the FFT U-Net paper:
    https://ieeexplore.ieee.org/document/9507635
    """
    ins = Input(input_shape)
    c1 = EncoderBlock(ins,n_filters)
    c2 = EncoderBlock(c1[0],n_filters*2)
    c3 = EncoderBlock(c2[0],n_filters*4)
    c4 = EncoderBlock(c3[0],n_filters*8)

    bottleneck =  Conv2D(n_filters*16,3, activation='relu',padding='same')(c4[1])
    bottleneck = BatchNormalization()(bottleneck)
    bottleneck = Conv2D(n_filters*16,3, activation='relu',padding='same')(bottleneck)
    bottleneck = BatchNormalization()(bottleneck)

    u1 = DecoderBlock(c4[0],bottleneck, n_filters*8)
    u2 = DecoderBlock(u1,c3[1], n_filters*4)
    u3 = DecoderBlock(u2,c2[1], n_filters*2)
    u4 = DecoderBlock(u3,c1[1], n_filters)
 
    conv9 = Conv2D(4, 3, activation='relu', padding='same')(u4)
    
    model = Model(inputs=[ins], outputs=[conv9])
    return model


# %% Models


class SAR_opt_Matcher():
    def __init__(self, **config):
        
        print("<SAR_opt_Matcher> instantiated")
        # ATTRIBUTES
        self.backbone = None
        self.n_filters = 0
        self.multiscale = None
        self.attention = None
        self.activation = None             
        self.model = None
            
    
    def set_attributes(self, config):
        
        self.backbone = config.get('backbone')
        self.n_filters = config.get('n_filters')
        self.multiscale = config.get('multiscale')
        self.attention = config.get('attention')
        self.activation = config.get('activation') 
        
    def print_attributes(self):
        """ Print class attrbutes """
        print("\n---Printing class attributes:---")
        for attribute, value in self.__dict__.items():
            if hasattr(value, "shape"):
                print(f"{attribute} = {value.shape}")
            else:
                print(f"{attribute} = {value}")
        print("\n")
        
    
    def export_attributes(self):        
        def to_dict(self):
            #Creates a dict of important attributes to export
            return {"backbone": self.backbone, 
                    "n_filters": self.n_filters, 
                    "multiscale": self.multiscale,
                    "attention" : self.attention,
                    "activation" : self.activation}
        
        config = to_dict(self)
        with open(self.model_name + '.json', 'w') as file:
            json.dump(config, file)
       
        
    def create_model(self, reference_im_shape = (256,256,1), floating_im_shape = (192,192,1), normalize = True, **config):
               
        self.backbone = config.get('backbone')
        self.n_filters = config.get('n_filters')
        self.multiscale = config.get('multiscale')
        self.attention = config.get('attention')
        self.activation = config.get('activation') 
        
        
        backbone = self.backbone
        n_filters = self.n_filters
        attention = self.attention
        multiscale = self.multiscale
        activation = self.activation
        
        # Assume shape given is in channel_last format, also assumes images are squares.
        # Define input shapes of input images
        opt_in = Input(shape = reference_im_shape)
        sar_in = Input(shape = floating_im_shape)
        float_im_size = floating_im_shape[0] - 1 
        response_crop = ((float_im_size,float_im_size),(float_im_size,float_im_size))
           
        # Instantiate backbone and extract feature maps
        if backbone.lower() == "marunet":
            heatmap = ResUNet_Attention((None,None,1), 
                                        n_filters = n_filters,
                                        attention = attention, 
                                        activation = activation)
                
            opt_heatmap = heatmap(opt_in)
            opt_heatmap = Conv2D(4, 3, activation = activation, padding='same', kernel_initializer='HeNormal', name = "psi_opt_o")(opt_heatmap)
            
            sar_heatmap = heatmap(sar_in)
            sar_heatmap = Conv2D(4, 3, activation = activation, padding='same', kernel_initializer='HeNormal', name = "psi_SAR_o")(sar_heatmap)
            
            loss = loss_module.crossEntropyNegativeMining
    
        elif backbone.lower() == "unet":
            heatmap = unet((None,None,1), n_filters = n_filters)
            opt_heatmap = heatmap(opt_in)
            sar_heatmap = heatmap(sar_in)
            loss = loss_module.crossEntropy
        
        else:
            raise Exception("Backbone not implemented")
    
    
        if multiscale:
            # Downscale input and extract features
            down_opt_heatmap = heatmap(MaxPooling2D((2,2))(opt_in))
            down_sar_heatmap = heatmap(MaxPooling2D((2,2))(sar_in))
            
            # Upsample and reduce channel dimension
            up_sar_heatmap = UpSampling2D((2,2),interpolation="bilinear")(down_sar_heatmap)
            up_opt_heatmap = UpSampling2D((2,2),interpolation="bilinear")(down_opt_heatmap)
            up_opt_heatmap = Conv2D(4, 3, activation = activation, padding='same',kernel_initializer='HeNormal', name = "psi_opt_d")(up_opt_heatmap)
            up_sar_heatmap = Conv2D(4, 3, activation = activation, padding='same',kernel_initializer='HeNormal', name = "psi_SAR_d")(up_sar_heatmap)
    
            # Multi-scale feature map
            opt_heatmap = concatenate([opt_heatmap,up_opt_heatmap], axis=3)
            sar_heatmap = concatenate([sar_heatmap,up_sar_heatmap], axis=3)
    
        # Reallign channels for FFT
        sar_heatmap = Permute((3,1,2))(sar_heatmap)
        opt_heatmap = Permute((3,1,2))(opt_heatmap)
    
        
        # FFT-based Cross Correlation
        xcorr = Lambda(loss_module.fft_layer)([opt_heatmap,sar_heatmap])
    
        if normalize:
            xcorr = Lambda(loss_module.Normalization_layer)([opt_heatmap, sar_heatmap, xcorr])
        
        # Crop the Normalized Cross Correlation heatmap so that matches correspond to origin (top-left corner) of the template
        out = Cropping2D(cropping=response_crop,data_format = "channels_first")(xcorr)
    
        # Move channels back to inital position
        out = Permute((2,3,1))(out)
    
        # Averagely reduce the channel number to 1, sharpen output if normalized
        if normalize:
            out = Lambda(lambda x: tf.divide(tf.reduce_mean(x,axis = 3,keepdims=True),1/30))(out) #1/30 is a temperature factor
        else:
            out = Lambda(lambda x: tf.reduce_mean(x,axis = 3,keepdims=True))(out)
    
        model = Model(inputs = [opt_in, sar_in], outputs = out)
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005)
        model.compile(optimizer = optimizer, loss = loss)
        
        # return model
        self.model = model
        
        
 
    def export_model_architecture(self):
        """ Plot model architecture (pdf is also possible)"""
        if self.model is not None:
            tf.keras.utils.plot_model (self.model, to_file = self.model_name + '.png', show_shapes = True, show_layer_names = True)



    def train(self, training_data : tf.data.Dataset, validation_data : tf.data.Dataset, epochs = 5):        
        if utils.confirmCommand("Train a new model?"):
            print("--training")
            my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5, verbose = 1), #15
                # tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 1),
                tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", 
                                                      factor = 0.5, 
                                                      patience = 3, 
                                                      min_lr = 0.00001, 
                                                      verbose = 1)] 
            
            history = self.model.fit(training_data, epochs = epochs,
                        validation_data = validation_data,
                        callbacks = my_callbacks)
            
            print("--Saving weights")
            self.model.save_weights("weights/" + self.model_name + ".h5")
            print("--Saving history")
            with open("weights/" + self.model_name + "_history", 'wb') as file:
                pickle.dump(history.history, file)
                
            self.export_attributes()
            
        
               
    def load_model(self):       
        # It seems like the support for exotic kinds of Python functions within the Lambda(...) 
        # doesn't play very nicely with Keras serialization.
        # So we only save the weights. Then we create the architecture from the code and load the weights. 
        
        print("--Loading")
        self.model_name = utils.open_file("Model to load:")
        
        # Find the .json file        
        with open(os.path.splitext(self.model_name)[0] + ".json", 'r') as f:
            config = json.load(f)
        
        # Set attributes
        self.set_attributes(config)
        
        # Create the architecture and load stored weights
        self.create_model(**config)
        self.model.load_weights(self.model_name)

                        
    
    def plot_history(self,):        
        with open(utils.open_file(self.model_name), 'rb') as f:
            history = pickle.load(f)    
        plt.plot(history['loss'])  
        plt.plot(history['val_loss'])
  
        
    def predict_heatmap(self, validation_data : tf.data.Dataset):
        if self.model is not None:
            print("--Calculating heatmaps")
            ncc_heatmap = self.model.predict(validation_data)
            if hasattr(ncc_heatmap, "shape"):
                if len(ncc_heatmap.shape) == 4:
                    ncc_heatmap = np.squeeze(ncc_heatmap, axis = -1)
                    
            return ncc_heatmap
        else:
            raise("Model not defined")
            
            
    def calculate_features(self, validation_data : tf.data.Dataset):
        """
        Create a model which output are the feature maps. 
        'matcher' is an instance of the <SarOptMatch.architectures.SAR_opt_Matcher> class
        """
        
        
        if self.model is not None:
            print("--Calculating feature maps")
            # Scan the matcher.model to find the right layers
            layer_names = [layer.name for layer in self.model.layers]
            
            psi_opt_o = layer_names.index("psi_opt_o")
            psi_sar_o = layer_names.index("psi_SAR_o")
            feature_maps = [self.model.layers[psi_opt_o].output, self.model.layers[psi_sar_o].output]
            if not self.multiscale: 
                print("Features: psi_opt_o and psi_SAR_o")
            if self.multiscale:
                psi_opt_d = layer_names.index("psi_opt_d")
                psi_sar_d = layer_names.index("psi_SAR_d")
                feature_maps = [self.model.layers[psi_opt_o].output, self.model.layers[psi_sar_o].output, 
                                self.model.layers[psi_opt_d].output, self.model.layers[psi_sar_d].output]             
                print("Features: psi_opt_o, psi_SAR_o, psi_opt_d,  psi_SAR_d")
            visualization_model = tf.keras.Model(inputs = self.model.input, outputs = feature_maps)
         
            feature_maps = visualization_model.predict(validation_data)
            return feature_maps
        else:
            raise("Model not defined")
        
        
            
        



















