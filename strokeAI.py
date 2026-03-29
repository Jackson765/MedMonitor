import tensorflow as tf

# 1. Load your trained model
model = tf.keras.models.load_model('strokeDetection.h5')
model.summary()

def detect(imgDirectory):
    img = tf.keras.utils.load_img(imgDirectory, target_size=(224, 224))

    # 2. Convert image to numpy array
    img_array = tf.keras.utils.img_to_array(img)

    # 3. Add a batch dimension (models expect [batch_size, height, width, channels])
    img_array = tf.expand_dims(img_array, 0)

    # 4. Normalize (if your model was trained with scaled pixels, e.g., 0-1)
    # img_array = img_array / 255.0

    return model.predict(img_array, verbose=0)[0][0]