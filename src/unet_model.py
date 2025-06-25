import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input
from tensorflow.keras import Model

def conv_block(input_tensor, num_filters):
    """Convolutional block: Conv2D -> ReLU -> Conv2D -> ReLU"""
    x = Conv2D(num_filters, 3, padding='same', activation='relu', kernel_initializer='he_normal')(input_tensor)
    x = Conv2D(num_filters, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    return x

def encoder_block(input_tensor, num_filters):
    """Encoder block: conv_block -> MaxPooling2D"""
    x = conv_block(input_tensor, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p # Return both the output of conv_block and pooling for skip connection

def decoder_block(input_tensor, skip_connection_tensor, num_filters):
    """Decoder block: Conv2DTranspose -> concatenate -> conv_block"""
    x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    x = concatenate([x, skip_connection_tensor]) # Skip connection
    x = conv_block(x, num_filters)
    return x

def unet_model(input_size=(128, 128, 3)):
    """
    Defines the U-Net model architecture.

    Args:
        input_size (tuple): Tuple (height, width, channels) of input images.

    Returns:
        tf.keras.Model: The compiled U-Net model.
    """
    inputs = Input(input_size)

    # Encoder (Downsampling Path)
    # Block 1
    c1, p1 = encoder_block(inputs, 16) # Output c1 for skip, p1 for next block
    # Block 2
    c2, p2 = encoder_block(p1, 32)
    # Block 3
    c3, p3 = encoder_block(p2, 64)
    # Block 4
    c4, p4 = encoder_block(p3, 128)

    # Bottleneck
    bottleneck = conv_block(p4, 256)

    # Decoder (Upsampling Path)
    # Block 4
    d4 = decoder_block(bottleneck, c4, 128)
    # Block 3
    d3 = decoder_block(d4, c3, 64)
    # Block 2
    d2 = decoder_block(d3, c2, 32)
    # Block 1
    d1 = decoder_block(d2, c1, 16)

    # Output layer (sigmoid for binary segmentation)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(d1) # 1 filter for binary mask

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

if __name__ == '__main__':
    # Test the model creation
    model = unet_model(input_size=(128, 128, 3))
    model.summary() # Print a summary of the model's layers and parameters
    print("\nU-Net model created successfully!")