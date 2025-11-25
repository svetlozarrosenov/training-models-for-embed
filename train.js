// train_1second_windows.js  – 100% работи с tfjs-node (2025)
import * as tf from '@tensorflow/tfjs-node';

const WINDOW_SIZE = 200;
const STEP = 25;

async function loadAndWindowCSV(filename, label) {
  console.log(`Зареждане на ${filename}...`);
  const csvDataset = tf.data.csv(`file://${process.cwd()}/${filename}`);
  const rawData = [];

  await csvDataset.forEachAsync(row => {
    rawData.push([
      parseFloat(row.x) / 1000.0,   // в g
      parseFloat(row.y) / 1000.0,
      parseFloat(row.z) / 1000.0
    ]);
  });

  const windows = [];
  const labels = [];

  for (let i = 0; i <= rawData.length - WINDOW_SIZE; i += STEP) {
    windows.push(rawData.slice(i, i + WINDOW_SIZE));
    labels.push(label);
  }

  console.log(`${filename} → ${windows.length} прозорци`);
  return {
    data: tf.tensor3d(windows),
    labels: tf.tensor1d(labels, 'int32')
  };
}

function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.conv1d({
    inputShape: [WINDOW_SIZE, 3],
    filters: 16,
    kernelSize: 9,
    padding: 'same',        // ← ТОВА Е КЛЮЧЪТ!
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling1d({ poolSize: 2 }));

  model.add(tf.layers.conv1d({
    filters: 32,
    kernelSize: 7,
    padding: 'same',        // пак 'same'
    activation: 'relu'
  }));
  model.add(tf.layers.globalAveragePooling1d());
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

async function main() {
  const running = await loadAndWindowCSV('running.csv', 1);
  const walking = await loadAndWindowCSV('walking.csv', 0);
  const resting = await loadAndWindowCSV('resting.csv', 0);
  const shaking = await loadAndWindowCSV('shaking.csv', 0);
  const tap = await loadAndWindowCSV('tap.csv', 0);

  const xs = tf.concat([running.data, walking.data, resting.data, shaking.data, tap.data]);
  const ys = tf.concat([running.labels, walking.labels, resting.labels, shaking.labels, tap.labels]);

  console.log(`\nОбщо прозорци: ${xs.shape[0]}`);
  console.log(`Форма: ${xs.shape}\n`);

  const model = createModel();
  model.summary();

  await model.fit(xs, ys, {
    epochs: 40,
    batchSize: 64,
    validationSplit: 0.15,
    shuffle: true,
    callbacks: tf.callbacks.earlyStopping({ monitor: 'val_accuracy', patience: 8 })
  });

  await model.save('file://./my-model');
  console.log('\nГОТОВО! Моделът е в ./my-model');
}

main().catch(console.error);