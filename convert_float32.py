import tensorflowjs as tfjs
import tensorflow as tf
import os

print("Зареждане на TF.js модел...")
model = tfjs.converters.load_keras_model('./my-model/model.json')

print("Конвертиране в чист TFLite (float32) – 100% съвместим с ESP32")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# НЕ слагаме никаква квантизация!
tflite_model = converter.convert()

with open('model_float32.tflite', 'wb') as f:
    f.write(tflite_model)

os.system("xxd -i model_float32.tflite > model_data.h")

size = os.path.getsize('model_float32.tflite') / 1024
print(f"ГОТОВО! model_float32.tflite ({size:.1f} KB) → готов за Heltec!")
