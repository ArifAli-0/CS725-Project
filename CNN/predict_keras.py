from keras.models import load_model

# Load the entire model
loaded_model = load_model('model.keras')

# Now you can use the loaded model for making predictions on new data

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('predict/download.jpeg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Make a prediction using the loaded model
result = loaded_model.predict(test_image)

# result will contain the prediction, e.g., [0.987] for real or [0.002] for fake
# You can set a threshold to classify it as real or fake
threshold = 0.5
if result[0][0] > threshold:
    prediction = 'real'
else:
    prediction = 'fake'

print(prediction)
