import * as tf from '@tensorflow/tfjs-node';

async function loadCSV(filename) {
  const csv = await tf.data.csv(`file://${process.cwd()}/${filename}`);
  const data = [];

  // Label is determined by filename: contains "running" → 1, otherwise → 0
  const isRunning = filename.toLowerCase().includes('running');

  await csv.forEachAsync(row => {
    const xs = [
      parseFloat(row.x) / 1000.0,
      parseFloat(row.y) / 1000.0,
      parseFloat(row.z) / 1000.0
    ];
    data.push(xs);
  });

  // Create labels array filled with 1 or 0 based on filename
  const labels = new Array(data.length).fill(isRunning ? 1 : 0);

  return {
    data: tf.tensor2d(data),
    labels: tf.tensor1d(labels, 'int32')
  };
}

async function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [3], units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

  model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

async function main() {
    console.log("Loading data...");
  
    const runningData = await loadCSV('running.csv');    // → label = 1 (тичане)
    const walkingData = await loadCSV('walking.csv');    // → label = 0 (ходене)
    const restingData = await loadCSV('resting.csv');    // → label = 0 (покой)
  
    console.log(`Running samples: ${runningData.data.shape[0]}`);
    console.log(`Walking samples: ${walkingData.data.shape[0]}`);
    console.log(`Resting samples: ${restingData.data.shape[0]}`);
  
    const xs = tf.concat([
      runningData.data,
      walkingData.data,
      restingData.data
    ]);
  
    const ys = tf.concat([
      runningData.labels,
      walkingData.labels,
      restingData.labels
    ]);
  
    const model = await createModel();
  
    console.log("Training model... (this may take 1–3 minutes)");
    await model.fit(xs, ys, {
      epochs: 60,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (epoch % 10 === 0 || epoch === 59) {
            console.log(`Epoch ${epoch + 1}/60 — loss: ${logs.loss.toFixed(4)} — accuracy: ${(logs.acc * 100).toFixed(2)}% — val_accuracy: ${(logs.val_acc * 100).toFixed(2)}%`);
          }
        }
      }
    });
  
    await model.save('file://./my-model');
    console.log("Model trained and saved to ./my-model/");
    console.log("Next steps:");
    console.log("  node convert.js   → creates .tflite");
    console.log("  node to-header.js → creates model_data.h");
    console.log("  Upload to Heltec → real-time running detection!");
}

main().catch(err => {
  console.error("Training failed:", err.message);
});