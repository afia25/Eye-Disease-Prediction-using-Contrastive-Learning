# Eye-Disease-Prediction-using-Contrastive-Learning

while adding GNN, why extractiong train features for only one epoch? while we usually extract ftr for 10 epochs.



Softmax is for classification, while reconstruction requires sigmoid to output pixel intensities.

However, the model performed badly when the
dimension of features is lower. We find that it is the initial
weight of the network that caused the problem. Therefore, our
next research object is to discover how to pre-train the initial
weight effectively. 

Medical datasets are expensive and, unfortunately, DL training needs a lot of labeled data samples. This is because most lesion sites in medical diagnosis have distinguishable characteristics. Self-supervised training (SSL) is effective to solve the lack of labeled data for neural networks by learning latent representation from unlabeled data p14. In this study, we proposed a deep learning model which can be trained in self-supervised manner and able to extract both local and global high level features and learn the correlation among features from the DR images. Our ambition is to inspire future research endeavors, ultimately leading to enhanced disease detection in medical imaging and thereby assisting healthcare professionals.



jei attention code gular acc valo ashbe (0.95) shei attention code guloke further improve korbo. by these ways:
*********** several ProCBAM modules are introduced into the ResNet18 network. that means adding several attention layers to the model
*********** shei attention gulor sathe further aro attention mechanism combine kora. jemon (paper 2) attention mechanism ProCBAM = CA + SA + SE




=========================================================================================================================================================================

give a moderate length of answer. not too long. not too short.

convVIT_tiny_pretext_manual_imageprocessor_batch_80_ep_10.keras



https://colab.research.google.com/drive/1JTKKMq0T228qE-woy7i54PVoBkbZhh1j?authuser=5#scrollTo=HLjhKoPuqLL1
https://colab.research.google.com/drive/13CS7-MCucxl4RAu9yT_7RJy8j9AOwO9L?authuser=2#scrollTo=HLjhKoPuqLL1
https://colab.research.google.com/drive/1TP5uagB-OvHmlELhJ4rAFyGxnhzOjfPg?authuser=5#scrollTo=HLjhKoPuqLL1
good result = https://colab.research.google.com/drive/1ZrBSAWvt6g8CfbJ3YcK1_004TW-9I6SM?authuser=5#scrollTo=HLjhKoPuqLL1  (not imp)
good result = https://colab.research.google.com/drive/1--5KCdbQoYzQJUdjlnt5lAQvWX4i2Zi0?authuser=5
good result = https://colab.research.google.com/drive/14dpN8kbqbk3L2ld4yYNU9yEY8xJsyLni?authuser=5#scrollTo=Pz8PorqG6uLj
good result = https://colab.research.google.com/drive/15DJec4BE6r4pke5QhWq6tjhEe1fCV3kx?authuser=2#scrollTo=Pz8PorqG6uLj

batch code
batch 20 =  https://colab.research.google.com/drive/1dQUXM_vEgmueyAtERPLNuCZ0W2wSpYi3?authuser=7#scrollTo=HLjhKoPuqLL1
batch 50 =  https://colab.research.google.com/drive/1Q-YKWXtR-OYxV_98mmE-IDvunqRXeD0e?authuser=5#scrollTo=9evidRponMWB  (K id 10)
batch 50 =  https://colab.research.google.com/drive/1tb5L7-zUihhb5K8R5WOOPF3semPlsg5O?authuser=5#scrollTo=HLjhKoPuqLL1  (K id 10)
batch 80 =  https://colab.research.google.com/drive/15DJec4BE6r4pke5QhWq6tjhEe1fCV3kx?authuser=2#scrollTo=HLjhKoPuqLL1
batch 150 = https://colab.research.google.com/drive/1odw0Rf-U7GBcUXgN1hYJX6TBuzRkx_Mg?authuser=2#scrollTo=lrCTJDIf9jgw
batch 200 = https://colab.research.google.com/drive/1heJ1edhzizQKvBz49nkOPUndl7WS3g_6?authuser=5#scrollTo=HLjhKoPuqLL1 

selected wgh = convVIT_tiny_pretext_manual_imageprocessor_batch_50_ep_70.keras



## Is it important to use image_processor(images, return_tensors="tf")["pixel_values"] for ConvNeXt? 
ans: Yes if you're using a pretrained ConvNeXt model, Use AutoImageProcessor to preprocess the image. bcs Hugging Face vision models like ConvNeXt are trained with very specific image preprocessing steps. image_processor will do ((x / 255) - mean) / std. If you're using image_processor, do not divide by 255 yourself. Let HuggingFace handle it by AutoImageProcessor. U DO NOT need AutoImageProcessor if you're using randomly initialized ConvNeXt (no wgh). 
 
=========================================================================================================================================================================
Show that using the contrastive learning pretrained weights as initialization gives better accuracy than using ImageNet pretrained weights.


ask chatgpt how to add cnn decoder after convvit encoder. 
go to this file and add a decoder = https://colab.research.google.com/drive/1Q-YKWXtR-OYxV_98mmE-IDvunqRXeD0e?authuser=5#scrollTo=-h0C3iwBDWkV
check autoencoder codes from KID = 1, 2.


###################################################################################################################################################################

## convVIT_tiny_pretext_manual_imageprocessor_batch_50_ep_70.keras
## unfrozen (trainable)
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.optimizers import Adam


# Downstream encoder - using pretraining_model.encoder for feature extraction
class ConvNeXtEncoder(tf.keras.Model):
    def __init__(self, pretraining_encoder, **kwargs):
        super(ConvNeXtEncoder, self).__init__(**kwargs)
        self.encoder = pretraining_encoder  # Use pretraining_model.encoder

    def call(self, inputs):
        # Ensure inputs have the shape (B, H, W, C)
        pixel_values = tf.transpose(inputs, perm=[0, 1, 2, 3])  # (B, 3, 224, 224)
        outputs = self.encoder(pixel_values)

        if isinstance(outputs, dict):
            if 'pooler_output' in outputs:
                return outputs['pooler_output']  # Shape: (B, hidden_dim)
            elif 'last_hidden_state' in outputs:
                return tf.reduce_mean(outputs['last_hidden_state'], axis=1)  # Mean pooling
        return outputs  # fallback


# Input layer
inputs = Input(shape=(224, 224, 3), name='pixel_values')

# Encoder
encoder = ConvNeXtEncoder(pretraining_encoder=pretraining_model.encoder)
features = encoder(inputs)  # shape: (B, hidden_dim)

# --- Classification Head ---
x_cls = layers.Dense(128, activation='relu')(features)
x_cls = layers.Dropout(0.2)(x_cls)
classification_output = layers.Dense(5, activation='softmax', name='classification_output')(x_cls)

# --- Decoder Head ---
# Project feature vector to a feature map
x = layers.Dense(7 * 7 * 128, activation='relu')(features)  # You can change this size as needed
x = layers.Reshape((7, 7, 128))(x)

x = layers.UpSampling2D((2, 2))(x)  # 7x7 → 14x14
x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)

x = layers.UpSampling2D((2, 2))(x)  # 14x14 → 28x28
x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)

x = layers.UpSampling2D((2, 2))(x)  # 28x28 → 56x56
x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)

x = layers.UpSampling2D((2, 2))(x)  # 56x56 → 112x112
x = layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)

x = layers.UpSampling2D((2, 2))(x)  # 112x112 → 224x224
decoder_output = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same', name='reconstruction_output')(x)

# Final model
model = Model(inputs=inputs, outputs=[classification_output, decoder_output])

# Compile model
model.compile(
    optimizer=Adam(),
    loss={
        'classification_output': 'categorical_crossentropy',
        'reconstruction_output': 'mse'
    },
    metrics={
        'classification_output': 'accuracy',
        'reconstruction_output': 'mse'
    }
)

# Train
history = model.fit(
    labeled_train_dataset,
    epochs=10,
    validation_data=val_dataset
)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
decoder



x = layers.Dense(64, activation='relu')(model.output)
classification_output = layers.Dense(5, activation='softmax')(x)
###final_model = Model(inputs=mobilenet.input, outputs=final_output)
encoder_model = Model(inputs=mobilenet.input, outputs=x)   # outputs=previous layer of classification_output. classification_output baad jabe




