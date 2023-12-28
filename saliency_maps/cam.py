import numpy as np
import cv2
from keras import backend as K
import matplotlib.cm as cm
from tensorflow.keras.utils import img_to_array, array_to_img

def get_class_activation_map(model, img,  last_conv_layer_name):
    ''' 
    this function computes the class activation map
    
    Inputs:
        1) model (tensorflow model) : trained model
        2) img (numpy array of shape (224, 224, 3)) : input image
    '''
    
    # expand dimension - create batch of one image
    img = np.expand_dims(img, axis=0)

    # predict the top class and get it's label
    predictions = model.predict(img, verbose=0)
    label_index = np.argmax(predictions)

    # Get the input weights to the softmax of all classes and then for the winning class.
    class_weights = model.layers[-1].get_weights()[0] # shape (num_of_neurons_in_dense, num_classes)

    class_weights_winner = class_weights[:, label_index] # (num_of_neurons_in_dense, )

    # Get last Convolutional layer
    final_conv_layer = model.get_layer(last_conv_layer_name)

    # Get all filters from last Conv layer (1, filt_size, filt_size, num_of_neurons)
    get_output = K.function([model.layers[0].input],[final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img])
    
    # Squeeze conv map to shape image to size (filt_size, filt_size, num_of_neurons)
    conv_outputs = np.squeeze(conv_outputs)

    # get class activation map for object class that is predicted to be in the image - multiply weights, maps and sum
    final_output = np.dot(cv2.resize(conv_outputs, dsize=(150, 150), interpolation=cv2.INTER_CUBIC), class_weights_winner).reshape(150,150) # dim: 224 x 224
    
    # return class activation map
    return final_output, label_index

def save_and_display_cam(img_array, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = img_array
    # img = keras.preprocessing.image.load_img(img_path)
    # img = keras.preprocessing.image.img_to_array(img)

    #img = cv2.resize(img, [150, 150])
    # Rescale heatmap to a range 0-255
    OldRange = (np.max(heatmap) - np.min(heatmap))  
    NewRange = 255 - 0
    heatmap = (((heatmap - np.min(heatmap)) * NewRange) / OldRange) + 0
    heatmap = np.uint8(heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    
    superimposed_img = array_to_img(superimposed_img)
    return superimposed_img

