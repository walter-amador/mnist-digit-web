const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const scaleFactor = 10;
let drawing = false;

ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);

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
  const outputNodes = document.querySelectorAll('.output-node');
  outputNodes.forEach((outputNode) => {
    outputNode.innerHTML = '';
  });
  const finalResult = document.querySelector('.final-result');
  if (finalResult) finalResult.remove();
});

document.getElementById('train').addEventListener('click', async () => {
  const numInput = preprocessCanvas();
  const inputNeurons = parseInt(document.getElementById('inputNeurons').value);
  const hiddenNeurons = parseInt(
    document.getElementById('hiddenNeurons').value
  );
  const outputNeurons = parseInt(
    document.getElementById('outputNeurons').value
  );
  const lr = parseFloat(document.getElementById('learningRate').value);
  const epochs = parseInt(document.getElementById('epochs').value);
  const labelPosition = parseInt(
    document.getElementById('positionLabel').value
  );
  const label = Array.from({ length: outputNeurons }, () => 0);
  label[labelPosition] = 1;

  console.log('numInput:', numInput.length);
  console.log('labelPosition:', labelPosition);
  console.log('label:', label);
  window.scrollTo(0, document.body.scrollHeight);
  const output = await trainModel(
    numInput,
    label,
    inputNeurons,
    hiddenNeurons,
    outputNeurons,
    lr,
    epochs
  );
  console.log('Output:', output);
  let prediction = output[0].indexOf(Math.max(...output[0]));
  console.log('Prediction:', prediction);
  let confidence = output[0][prediction] * 100;
  console.log(`Confidence: ${confidence.toFixed(2)}%`);
  const nnContainer = document.querySelector('.nn-container');
  const finalResult = document.createElement('div');
  finalResult.classList.add('final-result');
  finalResult.innerHTML = `Prediction: ${prediction} Confidence: ${confidence.toFixed(
    2
  )}%`;
  nnContainer.appendChild(finalResult);
});

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

// Sigmoid function
const sigmoid = (x) =>
  x.map((row) => row.map((value) => 1 / (1 + Math.exp(-value))));

// Derivative of sigmoid function
const sigmoidDerivative = (x) =>
  x.map((row) => row.map((value) => value * (1 - value)));

// Matrix operations
const dot = (a, b) => {
  return a.map((row) =>
    b[0].map((_, i) =>
      row.reduce((sum, val, j) => sum + val * (b[j]?.[i] || 0), 0)
    )
  );
};

const transpose = (m) => m[0].map((_, i) => m.map((row) => row[i]));
const add = (a, b) => a.map((row, i) => row.map((val, j) => val + b[i][j]));
const subtract = (a, b) =>
  a.map((row, i) => row.map((val, j) => val - b[i][j]));
const multiplyScalar = (m, scalar) =>
  m.map((row) => row.map((val) => val * scalar));
const sumColumns = (m) => [
  m[0].map((_, i) => m.reduce((sum, row) => sum + row[i], 0)),
];
const elementWiseMultiply = (a, b) =>
  a.map((row, i) => row.map((val, j) => val * b[i][j]));

// sleep function
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const trainModel = async (
  numInput,
  label,
  inputNeurons,
  hiddenNeurons,
  outputNeurons,
  lr,
  epochs
) => {
  // Reshape numInput and label to match the Python code
  numInput = [numInput];
  label = [label];

  // Weights and biases
  let wh = Array.from({ length: inputNeurons }, () =>
    Array.from({ length: hiddenNeurons }, () => Math.random())
  );
  let bh = [Array.from({ length: hiddenNeurons }, () => Math.random())];
  let wout = Array.from({ length: hiddenNeurons }, () =>
    Array.from({ length: outputNeurons }, () => Math.random())
  );
  let bout = [Array.from({ length: outputNeurons }, () => Math.random())];
  let output = [];
  for (let i = 0; i < epochs; i++) {
    const demoSpeed = parseInt(document.getElementById('demoSpeed').value);
    await sleep(demoSpeed);
    // Forward propagation
    let hiddenLayerInput = add(dot(numInput, wh), bh);
    let hiddenLayerOutput = sigmoid(hiddenLayerInput);

    let outputLayerInput = add(dot(hiddenLayerOutput, wout), bout);
    output = sigmoid(outputLayerInput);
    assignOutput(output);

    // Backward propagation
    let E = subtract(label, output);
    let dOutput = elementWiseMultiply(E, sigmoidDerivative(output)); // element-wise multiplication here
    let errorAtHiddenLayer = dot(dOutput, transpose(wout));
    let dHiddenLayer = elementWiseMultiply(
      errorAtHiddenLayer,
      sigmoidDerivative(hiddenLayerOutput)
    ); // and here

    // Update weights and biases
    wout = add(
      wout,
      multiplyScalar(dot(transpose(hiddenLayerOutput), dOutput), lr)
    );
    assignWeights(wout);
    bout = add(bout, multiplyScalar(sumColumns(dOutput), lr));
    wh = add(wh, multiplyScalar(dot(transpose(numInput), dHiddenLayer), lr));
    bh = add(bh, multiplyScalar(sumColumns(dHiddenLayer), lr));
  }
  return output;
};

function assignWeights(weights) {
  weights = weights.flat();
  const outputNodes = document.querySelectorAll('.output-node');
  let divWeights = [];
  outputNodes.forEach((outputNode) => {
    const divWeight = document.createElement('div');
    divWeight.classList.add('weights');
    divWeights.push(divWeight);
    outputNode.appendChild(divWeight);
  });

  weights.forEach((weight, i) => {
    const neuron = Math.floor(i / 2);
    divWeights[neuron].innerHTML = `${
      divWeights[neuron].innerHTML
    }<span class="weight">${weight.toFixed(2)}</span>`;
  });
}

function assignOutput(output) {
  output = output.flat();
  const outputNodes = document.querySelectorAll('.output-node');
  outputNodes.forEach((outputNode, i) => {
    outputNode.innerHTML = `<span class="output">${output[i].toFixed(
      2
    )}</span>`;
  });
}

// Draw Neural Network

function createConnection(node1, node2) {
  const container = document.querySelector('.nn-container');
  const containerRect = container.getBoundingClientRect();

  const rect1 = node1.getBoundingClientRect();
  const rect2 = node2.getBoundingClientRect();

  // Adjust coordinates relative to the container
  const x1 = rect1.left + rect1.width / 2 - containerRect.left;
  const y1 = rect1.top + rect1.height - containerRect.top;
  const x2 = rect2.left + rect2.width / 2 - containerRect.left;
  const y2 = rect2.top - containerRect.top;

  const svg = document.getElementById('connections');
  const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  line.setAttribute('x1', x1);
  line.setAttribute('y1', y1);
  line.setAttribute('x2', x2);
  line.setAttribute('y2', y2);
  svg.appendChild(line);
}

document.querySelector('#outputNeurons').addEventListener('change', (e) => {
  const outputNeurons = parseInt(e.target.value);
  const hiddenNeurons = parseInt(
    document.getElementById('hiddenNeurons').value
  );
  showNeuralNetwork(outputNeurons, hiddenNeurons);
});
document.querySelector('#hiddenNeurons').addEventListener('change', (e) => {
  const hiddenNeurons = parseInt(e.target.value);
  const outputNeurons = parseInt(
    document.getElementById('outputNeurons').value
  );
  showNeuralNetwork(outputNeurons, hiddenNeurons);
});

const showNeuralNetwork = (outputNeurons, hiddenNeurons) => {
  const outputLayer = document.querySelector('.output-layer');
  outputLayer.innerHTML = '';
  for (let i = 0; i < outputNeurons; i++) {
    outputLayer.innerHTML += `<div class="node output-node"></div>`;
  }
  const hiddenLayer = document.querySelector('.hidden-layer');
  hiddenLayer.innerHTML = '';
  for (let i = 0; i < hiddenNeurons; i++) {
    hiddenLayer.innerHTML += `<div class="node hidden-node"></div>`;
  }
  const outputNodes = document.querySelectorAll('.output-node');
  const inputNodes = document.querySelectorAll('.hidden-node');
  const svg = document.getElementById('connections');
  svg.style.pointerEvents = 'none';
  svg.innerHTML = '';
  outputNodes.forEach((outputNode) => {
    inputNodes.forEach((inputNode) => createConnection(inputNode, outputNode));
  });

  const container = document.querySelector('.canvas-container');
  inputNodes.forEach((inputNode) => createConnection(container, inputNode));
};

const outputNeurons = parseInt(document.getElementById('outputNeurons').value);
const hiddenNeurons = parseInt(document.getElementById('hiddenNeurons').value);
showNeuralNetwork(outputNeurons, hiddenNeurons);
