const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const scaleFactor = 10;
let drawing = false;
let tfliteModel;
let tfLiteModelM3;
let tfLiteModelM5;
let tfLiteModelM7;

ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);

async function loadModels() {
  try {
    const responses = await Promise.all([
      fetch('assets/M3.tflite'),
      fetch('assets/M5.tflite'),
      fetch('assets/M7.tflite'),
    ]);
    const buffers = await Promise.all(
      responses.map((res) => res.arrayBuffer())
    );
    tfLiteModelM3 = await tflite.loadTFLiteModel(buffers[0]);
    tfLiteModelM5 = await tflite.loadTFLiteModel(buffers[1]);
    tfLiteModelM7 = await tflite.loadTFLiteModel(buffers[2]);
    console.log('Model Loaded!');
  } catch (error) {
    console.error('Error loading model', error);
  }
}
loadModels();

canvas.addEventListener('mousedown', () => (drawing = true));
canvas.addEventListener('mouseup', () => (drawing = false));
canvas.addEventListener('mouseleave', () => (drawing = false));
canvas.addEventListener('mousemove', draw);

// Touch support
canvas.addEventListener('touchstart', (e) => {
  e.preventDefault();
  drawing = true;
  const rect = canvas.getBoundingClientRect();
  const touch = e.touches[0];
  const x = Math.floor((touch.clientX - rect.left) / scaleFactor);
  const y = Math.floor((touch.clientY - rect.top) / scaleFactor);
  ctx.fillStyle = 'white';
  ctx.fillRect(x * scaleFactor, y * scaleFactor, scaleFactor, scaleFactor);
});
canvas.addEventListener('touchmove', (e) => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const touch = e.touches[0];
  const event = {
    clientX: touch.clientX,
    clientY: touch.clientY,
  };
  draw(event);
});
canvas.addEventListener('touchend', () => (drawing = false));
canvas.addEventListener('touchcancel', () => (drawing = false));

function draw(e) {
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();

  const x = Math.floor((e.clientX - rect.left) / scaleFactor);
  const y = Math.floor((e.clientY - rect.top) / scaleFactor);

  ctx.fillStyle = 'red';
  ctx.fillRect(x * scaleFactor, y * scaleFactor, scaleFactor, scaleFactor);

  ctx.fillStyle = 'white';
  ctx.fillRect(x * scaleFactor, y * scaleFactor, scaleFactor, scaleFactor);
}

document.getElementById('clearCanvas').addEventListener('click', () => {
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById('result-m3').textContent = '';
  document.getElementById('confidence-m3').textContent = '';
  document.getElementById('result-m5').textContent = '';
  document.getElementById('confidence-m5').textContent = '';
  document.getElementById('result-m7').textContent = '';
  document.getElementById('confidence-m7').textContent = '';
  document.getElementById('final-result').textContent = '';
  const previews = document.querySelectorAll('.preview');
  previews.forEach((preview) => {
    preview.innerHTML = '';
  });
});

document.getElementById('predict').addEventListener('click', async () => {
  if (!tfLiteModelM3 || !tfLiteModelM5 || !tfLiteModelM7) {
    alert('Models not loaded yet!');
    return;
  }
  document.getElementById('final-result').scrollIntoView({ behavior: 'smooth' });
  layersWorking();

  setTimeout(() => {
    const imageData = preprocessCanvas();
    const inputTensor = tf.tensor(imageData, [1, 28, 28, 1]);
    const outputM3 = tfLiteModelM3.predict(inputTensor);
    const outputM5 = tfLiteModelM5.predict(inputTensor);
    const outputM7 = tfLiteModelM7.predict(inputTensor);
    const predictionsM3 = outputM3.dataSync();
    const predictionsM5 = outputM5.dataSync();
    const predictionsM7 = outputM7.dataSync();

    const predictedLabelM3 = predictionsM3.indexOf(Math.max(...predictionsM3));
    const predictedLabelM5 = predictionsM5.indexOf(Math.max(...predictionsM5));
    const predictedLabelM7 = predictionsM7.indexOf(Math.max(...predictionsM7));

    const confidenceM3 = (predictionsM3[predictedLabelM3] * 100).toFixed(2);
    const confidenceM5 = (predictionsM5[predictedLabelM5] * 100).toFixed(2);
    const confidenceM7 = (predictionsM7[predictedLabelM7] * 100).toFixed(2);

    const finalLabel = getFinalPrediction(
      predictedLabelM3,
      confidenceM3,
      predictedLabelM5,
      confidenceM5,
      predictedLabelM7,
      confidenceM7
    );

    document.getElementById('result-m3').textContent = predictedLabelM3;
    document.getElementById('confidence-m3').textContent = `${confidenceM3}%`;
    document.getElementById('result-m5').textContent = predictedLabelM5;
    document.getElementById('confidence-m5').textContent = `${confidenceM5}%`;
    document.getElementById('result-m7').textContent = predictedLabelM7;
    document.getElementById('confidence-m7').textContent = `${confidenceM7}%`;
    document.getElementById('final-result').textContent = finalLabel;
    layersWorking();
  }, 3000);
});

function layersWorking() {
  const layers = document.querySelectorAll('.model-layer');
  layers.forEach((layer) => {
    layer.classList.toggle('loading');
  });
}

function getFinalPrediction(
  predictedLabelM3,
  confidenceM3,
  predictedLabelM5,
  confidenceM5,
  predictedLabelM7,
  confidenceM7
) {
  const votes = [predictedLabelM3, predictedLabelM5, predictedLabelM7];
  const confidence = {
    [predictedLabelM3]: confidenceM3,
    [predictedLabelM5]: confidenceM5,
    [predictedLabelM7]: confidenceM7,
  };

  const voteCounts = votes.reduce((acc, label) => {
    acc[label] = (acc[label] || 0) + 1;
    return acc;
  }, {});

  let finalPrediction = null;
  let maxVotes = 0;
  for (const [label, count] of Object.entries(voteCounts)) {
    if (count > maxVotes) {
      maxVotes = count;
      finalPrediction = label;
    }
  }

  if (maxVotes === 1) {
    finalPrediction = Object.keys(confidence).reduce((bestLabel, label) =>
      confidence[label] > confidence[bestLabel] ? label : bestLabel
    );
  }

  return finalPrediction;
}

function preprocessCanvas() {
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = 28;
  tempCanvas.height = 28;
  const tempCtx = tempCanvas.getContext('2d');

  // Draw the main canvas onto the temporary canvas
  tempCtx.drawImage(canvas, 0, 0, 28, 28);

  tempCanvas.style.border = '1px solid red';

  const imgData = tempCtx.getImageData(0, 0, 28, 28);
  // Visual check: append the tempCanvas to the document to verify its contents
  const previews = document.querySelectorAll('.preview');
  const dataURL = tempCanvas.toDataURL();
  previews.forEach((preview) => {
    const img = new Image();
    img.src = dataURL;
    preview.appendChild(img);
  });
  let grayscaleData = [];

  for (let i = 0; i < imgData.data.length; i += 4) {
    const r = imgData.data[i];
    const g = imgData.data[i + 1];
    const b = imgData.data[i + 2];
    const avg = (r + g + b) / 3;

    grayscaleData.push(avg / 255);
  }

  return grayscaleData;
}
