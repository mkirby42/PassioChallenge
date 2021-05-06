import os
import tensorflow as tf

cwd = os.getcwd()

mobile_net = tf.keras.applications.MobileNetV2(
    input_shape = (224, 224, 3),
    include_top = True,
    weights = "imagenet",
)

l2_reg = tf.keras.regularizers.l2(0.0001)

for layer in mobile_net.layers:
    for attr in ['kernel_regularizer']:
      # If layer has regularization attribute add l2
      if hasattr(layer, attr):
        setattr(layer, attr, l2_reg)

# Since changing attributes just modifies the model config we need to save the weights and reload the model
model_json = mobile_net.to_json()
mobile_net.save_weights(f"{cwd}/weights.h5")
reconstructed_mobile_net = tf.keras.models.model_from_json(model_json)
reconstructed_mobile_net.load_weights(f"{cwd}/weights.h5", by_name = True)

# Save model to the challenge directory
reconstructed_mobile_net.save(f"{cwd}/mobileNetV2+L2Reg.h5", save_format = "h5")
