import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np
import random
import pickle
import SarOptMatch


# Assign seed
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


if not ("dataset_loaded" in locals()):
    dataset_loaded = False
    if not dataset_loaded:
        """ Generate dataset """
        sar_files, opt_files, offsets = SarOptMatch.dataset.sen1_2(data_path = 'C:/Users/miche/Downloads/SEN1-2', 
                                                                seed = seed, 
                                                                ims_per_folder = 1) #100
        """ Process and split dataset """
        # training_data and validation_data are used for analytics
        # validation_dataRGB only for visualization
        training_data, validation_data, validation_dataRGB = SarOptMatch.dataset.split_data(sar_files, opt_files, offsets, 
                                                                        batch_size = 4, 
                                                                        seed = seed, 
                                                                        masking_strategy = "unet")   
    dataset_loaded = True
    del training_data




train = False        
""" Train or load a model """ 
if not ('matcher' in locals()):
    matcher = SarOptMatch.architectures.SAR_opt_Matcher()  
    matcher.print_attributes()
    
    if train:
        config = {'model_name' : "marunet_vanilla",
                  'backbone' : 'marunet',
                  'n_filters' : 32,
                  'multiscale' : True,
                  'attention' : True, 
                  'activation' : "elu"
                  }
        matcher.create_model(**config)
        matcher.train(training_data, validation_data, epochs = 5)
    else:
        matcher.load_model()


heatmaps = matcher.predict_heatmap(validation_data)
# np.save("heatmap.npy", heatmaps)
# heatmaps = np.load("heatmap.npy")

    

""" Feature maps calculations"""

feature_maps = matcher.calculate_features(validation_data)


""" Visualization"""
SarOptMatch.visualization.visualize_dataset_with_GUI(validation_dataRGB, heatmaps, feature_maps)


error = SarOptMatch.evaluation.print_results(validation_data, heatmaps)    

    
    
    





















