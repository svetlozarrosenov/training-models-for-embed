import * as tf from '@tensorflow/tfjs-node';

const WINDOW_SIZE = 200;  // ← трябва да е точно 100
const STEP = 25;

async function testCSV(filename) {
  console.log(`Loading data from ${filename}...`);
  const csv = await tf.data.csv(`file://${process.cwd()}/${filename}`);
  const data = [];

  await csv.forEachAsync(row => {
    const xs = [
      parseFloat(row.x) / 1000.0,
      parseFloat(row.y) / 1000.0,
      parseFloat(row.z) / 1000.0
    ];
    data.push(xs);
  });

  const windows = [];
  for (let i = 0; i <= data.length - WINDOW_SIZE; i += STEP) {
    windows.push(data.slice(i, i + WINDOW_SIZE));
  }
  const tensor = tf.tensor3d(windows);

  console.log(`Loaded ${data.length} samples → ${windows.length} windows. Loading model...`);

  const model = await tf.loadLayersModel('file://./my-model/model.json');
  console.log("Model loaded. Running inference...");

  const predictions = model.predict(tensor);
  const results = await predictions.array();

  let runningCount = 0;
  results.forEach((prob, i) => {
    const isRunning = prob[0] > 0.7;
    if (isRunning) runningCount++;

    if (i < 20 || i % 1000 === 0 || i === results.length - 1) {
      console.log(`Row ${i}: probability = ${(prob[0] * 100).toFixed(2)}% → ${isRunning ? "RUNNING" : "NOT running"}`);
    }
  });

  const runningPercent = ((runningCount / results.length) * 100).toFixed(2);
  const notRunningPercent = ((1 - runningCount / results.length) * 100).toFixed(2);

  console.log("\nDONE!");
  console.log(`Total windows: ${results.length}`);        // ← промених само текста тук
  console.log(`RUNNING: ${runningCount} (${runningPercent}%)`);
  console.log(`NOT running: ${results.length - runningCount} (${notRunningPercent}%)`);

  if (runningCount / results.length > 0.6) {
    console.log("\nVERDICT: THE DOG IS RUNNING!!!");
  } else {
    console.log("\nVERDICT: The dog is calm.");
  }
}

const filename = process.argv[2] || 'test_while_running.csv';
testCSV(filename).catch(err => {
  console.error("Error:", err.message);
});