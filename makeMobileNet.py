import os
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K

cwd = os.getcwd()

mobile_net = tf.keras.applications.MobileNetV2(
    input_shape = (224, 224, 3),
    include_top = True,
    weights = "imagenet",
)

l2_norm = Lambda(lambda  x: K.l2_normalize(x ,axis = 1))(mobile_net.layers[-1].output)
mobileNet_L2 = Model(inputs = mobile_net.inputs, outputs = l2_norm)

# Save model to the challenge directory
mobileNet_L2.save(f"{cwd}/mobileNetV2_L2.h5", save_format = "h5")
