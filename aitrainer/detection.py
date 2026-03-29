import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

plt.rc('image', cmap='gray')
plt.rc('figure', autolayout=True)

def simplifyImage(images):
    image_path = "Image Path"

    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=1)  
    image = tf.image.resize(image, [300, 300])
    image = tf.image.convert_image_dtype(image, tf.float32)

    print("Original Image Shape:", image.shape)

    plt.figure(figsize=(5,5))
    plt.imshow(tf.squeeze(image))
    plt.title("Original Image")
    plt.axis('off')
    plt.show()

    # Add batch dimension
    image = tf.expand_dims(image, axis=0)
    kernel = tf.constant([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=tf.float32)

    kernel = tf.reshape(kernel, [3, 3, 1, 1])

    conv_output = tf.nn.conv2d(
        input=image,
        filters=kernel,
        strides=1,
        padding='SAME'
    )

    print("After Convolution Shape:", conv_output.shape)

    plt.figure(figsize=(5,5))
    plt.imshow(tf.squeeze(conv_output))
    plt.title("After Convolution")
    plt.axis('off')
    plt.show()
    relu_output = tf.nn.relu(conv_output)

    print("After ReLU Shape:", relu_output.shape)

    plt.figure(figsize=(5,5))
    plt.imshow(tf.squeeze(relu_output))
    plt.title("After ReLU Activation")
    plt.axis('off')
    plt.show()

    pool_output = tf.nn.max_pool2d(
        input=relu_output,
        ksize=2,
        strides=2,
        padding='SAME'
    )

    print("After Pooling Shape:", pool_output.shape)

    plt.figure(figsize=(5,5))
    plt.imshow(tf.squeeze(pool_output))
    plt.title("After Max Pooling")
    plt.axis('off')
    plt.show()
    flatten_layer = tf.keras.layers.Flatten()
    flatten_output = flatten_layer(pool_output)

    print("After Flatten Shape:", flatten_output.shape)

    print("First 20 Flattened Values:")
    print(flatten_output.numpy()[0][:20])
    dense_layer = tf.keras.layers.Dense(
        units=64,         
        activation='relu' 
    )

    dense_output = dense_layer(flatten_output)

    print("After Fully Connected Layer Shape:", dense_output.shape)

# 1. Load images from your directory
train_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/training',
    image_size=(224, 224), # Standard size for models like MobileNet
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/validation',
    image_size=(224, 224),
    batch_size=32
)

base_model = tf.keras.applications.EfficientNetB0(input_shape=(224, 224, 3), include_top=False)
base_model.trainable = False # Freeze the base to start

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid') # Sigmoid for binary (Fall/No Fall)
])

# 3. Compile and Train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=10)

# 4. Save for your app
model.save('fallDetection.h5')