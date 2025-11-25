# convert.py
import tensorflowjs as tfjs
import tensorflow as tf

print("Зареждане на модела...")
model = tfjs.converters.load_keras_model('./my-model/model.json')
print(f"Готово! Вход: {model.input_shape} → Изход: {model.output_shape}")

print("Конвертиране в TFLite с максимална оптимизация...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # най-доброто за малки модели

tflite_model = converter.convert()

with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

print("УСПЕХ! model_quantized.tflite е създаден")
print("Сега превръщаме в C-масив за Heltec...")
import os
os.system("xxd -i model_quantized.tflite > model_data.h")

import os
size = os.path.getsize('model_quantized.tflite') / 1024
print(f"Размер на модела: {size:.1f} KB → идеален за Heltec Wireless Tracker!")