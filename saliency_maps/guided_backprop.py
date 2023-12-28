import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
import numpy as np


H, W = 150, 150
# Reference: https://github.com/eclique/keras-gradcam/blob/master/grad_cam.py

def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

    from tensorflow.keras.models import Model

@tf.custom_gradient
def guidedRelu(x):
  def grad(dy):
    return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
  return tf.nn.relu(x), grad

def create_saliency_map(model, layer_name, img_array):
  gb_model = Model(
      inputs = [model.inputs],
      outputs = [model.get_layer(layer_name).output]
  )
  layer_dict = [layer for layer in gb_model.layers[1:] if hasattr(layer,'activation')]
  for layer in layer_dict:
    if layer.activation == tf.keras.activations.relu:
      layer.activation = guidedRelu

  with tf.GradientTape() as tape:
    inputs = tf.cast(np.array([img_array]), tf.float32)
    tape.watch(inputs)
    outputs = gb_model(inputs)

  return tape.gradient(outputs,inputs)[0]

  # import tensorflow as tf

# @tf.RegisterGradient("GuidedRelu")
# def _GuidedReluGrad(op, grad):
#    gate_f = tf.cast(op.outputs[0] > 0, "float32") #for f^l > 0
#    gate_R = tf.cast(grad > 0, "float32") #for R^l+1 > 0
#    return gate_f * gate_R * grad

# model = modelCNN
# with tf.compat.v1.get_default_graph().gradient_override_map({'Relu': 'GuidedRelu'}):
#   gb_model = Model(
#       inputs = [model.inputs],
#       outputs = [model.get_layer("conv2d_3").output]
#   )
  
#   with tf.GradientTape() as tape:
#     inputs = tf.cast(img_array, tf.float32)
#     tape.watch(inputs)
#     outputs = gb_model(inputs)

# grads = tape.gradient(outputs,inputs)[0]  