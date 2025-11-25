import * as tf from '@tensorflow/tfjs-node';

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

  const tensor = tf.tensor2d(data);
  console.log(`Loaded ${data.length} samples. Loading model...`);

  const model = await tf.loadLayersModel('file://./my-model/model.json');
  console.log("Model loaded. Running inference...");

  const predictions = model.predict(tensor);
  const results = await predictions.array();

  let runningCount = 0;
  results.forEach((prob, i) => {
    const isRunning = prob[0] > 0.7;  // you can adjust threshold: 0.6–0.8
    if (isRunning) runningCount++;

    // Print first 20, every 1000th, and the last one
    if (i < 20 || i % 1000 === 0 || i === results.length - 1) {
      console.log(`Row ${i}: probability = ${(prob[0] * 100).toFixed(2)}% → ${isRunning ? "RUNNING" : "NOT running"}`);
    }
  });

  const runningPercent = ((runningCount / results.length) * 100).toFixed(2);
  const notRunningPercent = ((1 - runningCount / results.length) * 100).toFixed(2);

  console.log("\nDONE!");
  console.log(`Total rows: ${results.length}`);
  console.log(`RUNNING: ${runningCount} (${runningPercent}%)`);
  console.log(`NOT running: ${results.length - runningCount} (${notRunningPercent}%)`);

  if (runningCount / results.length > 0.6) {
    console.log("\nVERDICT: THE DOG IS RUNNING!!!");
  } else {
    console.log("\nVERDICT: The dog is calm.");
  }
}

const filename = process.argv[2] || 'test_viki_running.csv';
testCSV(filename).catch(err => {
  console.error("Error:", err.message);
});