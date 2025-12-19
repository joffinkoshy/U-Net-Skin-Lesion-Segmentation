import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input, Dropout, BatchNormalization
from tensorflow.keras import Model
import tensorflow_addons as tfa

def conv_block(input_tensor, num_filters, dropout_rate=0.2):
    """Enhanced convolutional block with batch normalization and dropout"""
    x = Conv2D(num_filters, 3, padding='same', kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = Dropout(dropout_rate)(x)

    x = Conv2D(num_filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    return x

def attention_gate(x, g, inter_channel):
    """Attention gate mechanism for focusing on relevant features"""
    theta_x = Conv2D(inter_channel, 1, strides=1, padding='same')(x)
    phi_g = Conv2D(inter_channel, 1, strides=1, padding='same')(g)
    f = tf.keras.activations.relu(theta_x + phi_g)
    psi_f = Conv2D(1, 1, strides=1, padding='same')(f)
    return tf.keras.activations.sigmoid(psi_f)

def attention_block(input_tensor, skip_connection_tensor, num_filters):
    """Attention block with skip connection"""
    g = Conv2D(num_filters, 1, strides=1, padding='same')(skip_connection_tensor)
    x = Conv2D(num_filters, 1, strides=1, padding='same')(input_tensor)

    # Apply attention gate
    attn = attention_gate(x, g, num_filters)
    x = x * attn

    # Concatenate with skip connection
    x = concatenate([x, skip_connection_tensor])
    return x

def encoder_block(input_tensor, num_filters):
    """Encoder block with residual connection"""
    x = conv_block(input_tensor, num_filters)
    p = MaxPooling2D((2, 2))(x)

    # Residual connection
    residual = Conv2D(num_filters, 1, strides=2, padding='same')(input_tensor)
    p = p + residual
    return x, p

def decoder_block(input_tensor, skip_connection_tensor, num_filters):
    """Decoder block with attention mechanism"""
    x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)

    # Apply attention block
    x = attention_block(x, skip_connection_tensor, num_filters)

    x = conv_block(x, num_filters)
    return x

def attention_unet_model(input_size=(128, 128, 3), dropout_rate=0.2):
    """
    Enhanced Attention U-Net model with residual connections and attention gates.

    Args:
        input_size (tuple): Tuple (height, width, channels) of input images.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        tf.keras.Model: The compiled Attention U-Net model.
    """
    inputs = Input(input_size)

    # Encoder (Downsampling Path)
    # Block 1
    c1, p1 = encoder_block(inputs, 16)
    # Block 2
    c2, p2 = encoder_block(p1, 32)
    # Block 3
    c3, p3 = encoder_block(p2, 64)
    # Block 4
    c4, p4 = encoder_block(p3, 128)

    # Bottleneck with attention
    bottleneck = conv_block(p4, 256, dropout_rate)

    # Decoder (Upsampling Path) with Attention
    # Block 4
    d4 = decoder_block(bottleneck, c4, 128)
    # Block 3
    d3 = decoder_block(d4, c3, 64)
    # Block 2
    d2 = decoder_block(d3, c2, 32)
    # Block 1
    d1 = decoder_block(d2, c1, 16)

    # Output layer (sigmoid for binary segmentation)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(d1)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

if __name__ == '__main__':
    # Test the model creation
    model = attention_unet_model(input_size=(128, 128, 3))
    model.summary()
    print("\nAttention U-Net model created successfully!")
