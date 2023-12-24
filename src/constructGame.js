// src/App.js
import React, { useEffect, useRef } from "react";
import { Container, Row, Col, Form, Button } from "react-bootstrap";
import { useState } from "react";
import Banana from "./banana.png";
import Orange from "./Orange.png";
import Apple from "./apple (1).png";

import { Dropdown } from "react-bootstrap"; // Import Bootstrap Dropdown component
import Swal from "sweetalert2";
import file_data from "./sample_data.txt";

class NeuralNetworkModel {
  constructor(
    inputSize,
    hiddenSize,
    outputSize,
    activationFunction,
    activationFunctionOutputLayer
  ) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    this.activationFunction = activationFunction;
    this.activationFunctionOutputLayer = activationFunctionOutputLayer;
    this.initializeWeights();
    this.initializeBiases();
  }
  initializeWeights() {
    this.input_weights = [];
    this.hidden_weights = [];

    //initilize weights for InputHidden
    for (let i = 0; i < this.inputSize; i++) {
      var nodeweight = [];

      for (let j = 0; j < this.hiddenSize; j++) {
        nodeweight.push(this.getRandomNumber(-1.2, +1.2));
      }
      this.input_weights.push([...nodeweight]);
    }

    //initilize weights for hiddenoutput
    for (let i = 0; i < this.hiddenSize; i++) {
      var nodeweight = [];
      for (let j = 0; j < this.outputSize; j++) {
        nodeweight.push(this.getRandomNumber(-1.2, +1.2));
      }
      this.hidden_weights.push([...nodeweight]);
    }
  }

  initializeBiases() {
    this.hidden_biases = [];
    this.output_biases = [];
    for (var i = 0; i < this.hiddenSize; i++) {
      this.hidden_biases.push(this.getRandomNumber(-1.2, +1.2));
    }
    for (var j = 0; j < this.outputSize; j++) {
      this.output_biases.push(this.getRandomNumber(-1.2, +1.2));
    }
  }

  train(input, desired, learningRate) {
    this.hiddenLayerOutput = this.calculateHiddenLayerOutput(input);
    this.output = this.calculateOutput(this.hiddenLayerOutput);
    this.outputError = this.calculateOutputError(desired, this.output);
    this.hiddenLayerError = this.calculateHiddenLayerError(
      this.outputError,
      this.hiddenLayerOutput
    );
    this.adjustWeights(
      input,
      this.hiddenLayerOutput,
      this.outputError,
      this.hiddenLayerError,
      learningRate
    );
  }

  calculateHiddenLayerOutput(input) {
    var hiddenLayerOutput = [];

    for (let j = 0; j < this.hiddenSize; j++) {
      let summationX = 0;

      for (let i = 0; i < this.inputSize; i++) {
        summationX += input[i] * this.input_weights[i][j];
      }
      summationX += this.hidden_biases[j];

      hiddenLayerOutput[j] = this.activateX(summationX);
    }

    return hiddenLayerOutput;
  }

  calculateOutput(hiddenLayerOutput) {
    var output = [];
    var inputVector = [];
    for (let j = 0; j < this.outputSize; j++) {
      let summationX = 0;
      for (let i = 0; i < this.hiddenSize; i++) {
        summationX += hiddenLayerOutput[i] * this.hidden_weights[i][j];
      }
      summationX += this.output_biases[j];
      inputVector.push(summationX);
      if (this.activationFunctionOutputLayer != "softmax") {
        output.push(
          this.activateX(summationX, this.activationFunctionOutputLayer)
        );
      } else if (this.activationFunctionOutputLayer == "softmax") {
        output = this.softmax(inputVector);
      }
    }
    return output;
  }
  softmax(x) {
    const expX = x.map(Math.exp);
    const sumExpX = expX.reduce((acc, val) => acc + val, 0);
    return expX.map((value) => value / sumExpX);
  }
  activateX(x, fnc = null) {
    if (fnc == null) {
      if (this.activationFunction == "sigmoid") {
        return 1 / (1.0 + Math.exp(-x));
      } else if (this.activationFunction == "tanh") {
        return Math.tanh(x);
      } else if (this.activationFunction == "relu") {
        return Math.max(0.01 * x, x);
      }
    }

    if (fnc == "softmax") {
      return this.softmax(x);
    } else if (fnc == "tanh") {
      return Math.tanh(x);
    } else {
      return 1 / (1.0 + Math.exp(-x));
    }
  }

  calculateOutputError(desired, output) {
    var outputError = [];
    for (let i = 0; i < this.outputSize; i++) {
      outputError.push(desired[i] - output[i]);
    }
    return outputError;
  }
  sigmoidDerivative(x) {
    const sig = this.sigmoid(x);
    return sig * (1 - sig);
  }
  tanhDerivative(x) {
    const tanhx = this.tanh(x);
    return 1 - tanhx * tanhx;
  }
  reluDerivative(x) {
    return x < 0 ? 0.01 : 1;
  }
  relu(x) {
    return Math.max(0.01 * x, x);
  }
  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }
  tanh(x) {
    return Math.tanh(x);
  }
  calculateHiddenLayerError(outputError, hiddenLayerOutput) {
    var hiddenLayerError = [];
    for (let i = 0; i < this.hiddenSize; i++) {
      let summationX = 0;
      for (let j = 0; j < this.outputSize; j++) {
        summationX += outputError[j] * this.hidden_weights[i][j];
      }
      if (this.activationFunction == "tanh") {
        hiddenLayerError.push(
          summationX * (1 - Math.pow(hiddenLayerOutput[i], 2))
        );
      } else if (this.activationFunction == "sigmoid") {
        hiddenLayerError.push(
          summationX * this.sigmoidDerivative(hiddenLayerOutput[i])
        );
      } else if (this.activationFunction == "relu") {
        hiddenLayerError.push(
          summationX * this.reluDerivative(hiddenLayerOutput[i])
        );
      }
    }
    return hiddenLayerError;
  }

  getRandomNumber = (min, max) => Math.random() * (max - min) + min;
  adjustWeights(
    input,
    hiddenLayerOutput,
    outputError,
    hiddenLayerError,
    learningRate
  ) {
    for (let i = 0; i < this.inputSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        this.input_weights[i][j] +=
          learningRate * input[i] * hiddenLayerError[j];
      }
    }
    this.adjustBiases(this.hidden_biases, hiddenLayerError, learningRate);

    for (let i = 0; i < this.hiddenSize; i++) {
      for (let j = 0; j < this.outputSize; j++) {
        this.hidden_weights[i][j] +=
          learningRate * hiddenLayerOutput[i] * outputError[j];
      }
    }
    this.adjustBiases(this.output_biases, outputError, learningRate);
  }

  adjustBiases(biases, errors, learningRate) {
    for (let i = 0; i < biases.length; i++) {
      biases[i] += learningRate * errors[i];
    }
  }

  calculate_mse(trainingSamples) {
    let sum_err = 0;
    let input = new Array(2);
    let desired = 0;

    trainingSamples.map((sampleData) => {
      input[0] = sampleData.sweetness;
      input[1] = sampleData.color;
      desired = sampleData.fruit;

      let output = this.calculateOutput(this.calculateHiddenLayerOutput(input));
      let outputError = this.calculateOutputError(desired, output);
      for (let i = 0; i < this.outputSize; i++) {
        sum_err += Math.pow(outputError[i], 2);
      }
    });
    let mse = sum_err / trainingSamples.length;

    return mse;
  }
  isCorrectOutput(input, desired) {
    let output = this.calculateOutput(this.calculateHiddenLayerOutput(input));
    let predictedIndex = 0;

    for (let i = 1; i < this.outputSize; i++) {
      if (output[i] > output[predictedIndex]) {
        predictedIndex = i;
      }
    }

    var index = desired.indexOf(1);

    return predictedIndex == index;
  }
  predict_output(input) {
    let output = this.calculateOutput(this.calculateHiddenLayerOutput(input));
    let predictedIndex = 0;

    for (let i = 1; i < this.outputSize; i++) {
      if (output[i] > output[predictedIndex]) {
        predictedIndex = i;
      }
    }
    switch (predictedIndex) {
      case 0:
        return "Apple";
      case 1:
        return "Banana";
      case 2:
        return "Orange";
      default:
        return "Unknown";
    }
  }
}
function ConstructGame() {
  const [images, setImages] = useState([]);
  const [selectedOption, setSelectedOption] = useState("Apple");
  const [sweetness, setSweetness] = useState(0.5);
  const [colorValue, setColorValue] = useState(1);
  const [fruitValue, setFruitValue] = useState([1, 0, 0]);
  const [dataSet, setDataSet] = useState([]);
  const [normalizedData, setNormalizedDate] = useState([]);
  const [neurons, setNeurons] = useState(4);
  const [activationFunction, setActiationFunction] = useState("sigmoid");
  const [activationFunctionOutputLayer, setActivationFunctionOutputLayer] =
    useState("sigmoid");

  const [learningRate, setLearningRate] = useState(0.1);
  const [epoch, setEpoch] = useState(100000);
  const [goal, setGoal] = useState(0.01);
  let neuralNetwork = useRef(null);
  const [error, setError] = useState(0);
  const [accuracy, setAccuracy] = useState(100);

  const handleSelect = (eventKey, event) => {
    // adjust the selected option when an option is clicked
    setSelectedOption(event.target.textContent);

    if (event.target.textContent == "Apple") {
      setFruitValue([1, 0, 0]);
      // console.log(event.target.textContent);
    }
    if (event.target.textContent == "Banana") {
      setFruitValue([0, 1, 0]);
      //  console.log(event.target.textContent);
    }
    if (event.target.textContent == "Orange") {
      setFruitValue([0, 0, 1]);
      //  console.log(event.target.textContent);
    }
  };

  useEffect(() => {}, []);

  function clear() {
    setDataSet([]);
    setNormalizedDate([]);
  }
  function normalize(i, Imin, Imax) {
    var Nmax = 1;
    var Nmin = 0;
    var IN = (i - Imin) * ((Nmax - Nmin) / (Imax - Imin)) + Nmin;

    return IN;
  }
  function deNormalize(ON, Omin, Omax) {
    var Nmax = 1;
    var Nmin = 0;
    var O = (ON - Nmin) * ((Omax - Omin) / (Nmax - Nmin)) + Omin;
    return O;
  }
  function handle_sweetness_change(event) {
    setSweetness(eval(event.target.value));
  }

  function handleColorChange(event) {
    if (event.target.value.toLowerCase() == "r") {
      setColorValue(1);
    }
    if (event.target.value.toLowerCase() == "y") {
      setColorValue(2);
    }
    if (event.target.value.toLowerCase() == "o") {
      setColorValue(3);
    }
  }
  function addDate() {
    setDataSet([
      ...dataSet,
      { sweetness: sweetness, color: colorValue, fruit: fruitValue },
    ]);

    setNormalizedDate([
      ...normalizedData,

      {
        sweetness: normalize(sweetness, 0, 1),
        color: normalize(colorValue, 1, 3),
        fruit: fruitValue,
      },
    ]);
    //console.log(dataSet);
  }

  // Example: Get a random number between -2.4/2 and 2.4/2
  const minRange = -(2.4 / 2);
  const maxRange = 2.4 / 2;
  function softmax(x) {
    const expX = x.map(Math.exp);
    const sumExpX = expX.reduce((acc, val) => acc + val, 0);
    return expX.map((value) => value / sumExpX);
  }

  function softmaxDerivative(inputVector, index) {
    const soft = softmax(inputVector);
    return soft[index] * (1 - soft[index]);
  }
  function readDataFromFile() {}

  function processData(data) {
    const lines = data.split("\n");
    let datafile = [];

    lines.forEach((line) => {
      const parts = line.split(" ");
      const value = parts[0];
      const color = parts[1];
      let colorValue = 0;
      if (color == "r") {
        colorValue = 1;
      }
      if (color == "y") {
        colorValue = 2;
      }
      if (color == "o") {
        colorValue = 3;
      }
      const fruit = parts.slice(2).join(" ");
      let fruitValue = 0;

      if (fruit.includes("Apple")) {
        fruitValue = [1, 0, 0];
        // console.log(event.target.textContent);
      }
      if (fruit.includes("Banana")) {
        fruitValue = [0, 1, 0];

        //  console.log(event.target.textContent);
      }
      if (fruit.includes("Orange")) {
        fruitValue = [0, 0, 1];
        //  console.log(event.target.textContent);
      }
      datafile.push({ sweetness: value, color: colorValue, fruit: fruitValue });
    });

    setDataSet([...datafile]);
  }
  function handleFile(event) {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();

      reader.onload = function (e) {
        const content = e.target.result;
        processData(content);
      };

      reader.readAsText(file);
    }
  }
  function testModel() {
    if (neurons === 0) {
      alert("You need to train the model first");
      return;
    }

    let predictedFruit = neuralNetwork.current.predict_output([
      sweetness,
      colorValue,
    ]);
    console.log(predictedFruit);
    let title;
    let imageurl;
    if (predictedFruit == "Apple") {
      title = "Apple";
      imageurl = Apple;
    } else if (predictedFruit == "Banana") {
      title = "Banana";
      imageurl = Banana;
    } else if (predictedFruit == "Orange") {
      title = "Orange";
      imageurl = Orange;
    }

    Swal.fire({
      title: title,
      imageAlt: "Custom Image Alt Text", // Replace with the alt text for your image
      showCloseButton: true,
      confirmButtonText: "Close",
    });

    // You can use the output as needed in your application
  }

  function trainModel() {
    var input = new Array(2);
    neuralNetwork.current = new NeuralNetworkModel(
      2,
      neurons,
      3,
      activationFunction,
      activationFunctionOutputLayer
    );
    var total_count = 0;
    var counter = 0;
    for (let epochNum = 0; epochNum < epoch; epochNum++) {
      for (let sampleData of dataSet) {
        total_count++;
        input[0] = sampleData.sweetness;
        input[1] = sampleData.color;
        let desired = sampleData.fruit;

        neuralNetwork.current.train(input, desired, learningRate);
        let correct = neuralNetwork.current.isCorrectOutput(input, desired);
        if (correct == true) {
          counter++;
        }
      }

      let totalError=0
      totalError= neuralNetwork.current.calculate_mse(dataSet);
      setError(totalError);

      if (totalError <= goal) {
        console.log("total", total_count);
        setAccuracy(counter / total_count);
        setError(totalError);
        break;
      }
    }
    setAccuracy((counter * 100) / total_count);
    Swal.fire("finished training");
  }

  return (
    <>
      <Container className="mt-4 stystem_container">
        <Row>
          <Col>
            <h1>Fruit Recognition System</h1>
            <Form>
              <Row>
                <Col md={3}>
                  <Form.Group controlId="sweetness">
                    <Form.Label>Sweetness</Form.Label>
                    <Form.Control
                      type="number"
                      defaultValue={0.5}
                      placeholder="Enter sweetness"
                      min={0}
                      max={1}
                      step={0.1}
                      onChange={handle_sweetness_change}
                    />
                  </Form.Group>
                </Col>
                <Col md={3}>
                  <Form.Group controlId="color">
                    <Form.Label>Color</Form.Label>
                    <Form.Control
                      defaultValue={"r"}
                      type="text"
                      step="0.1"
                      placeholder="Enter color"
                      onChange={handleColorChange}
                    />
                  </Form.Group>
                </Col>

                <Col md={3}>
                  <Form.Label>Fruit</Form.Label>
                  <Dropdown onSelect={handleSelect} className="drown_down">
                    <Dropdown.Toggle
                      variant="success"
                      id="dropdown-basic"
                      style={{ width: "150px" }}
                    >
                      {selectedOption}
                    </Dropdown.Toggle>

                    <Dropdown.Menu>
                      <Dropdown.Item
                        eventKey="1"
                        href="#apple"
                        style={{ width: "150px" }}
                      >
                        Apple
                      </Dropdown.Item>
                      <Dropdown.Item
                        eventKey="2"
                        href="#banana"
                        style={{ width: "150px" }}
                      >
                        Banana
                      </Dropdown.Item>
                      <Dropdown.Item
                        eventKey="3"
                        href="#orange"
                        style={{ width: "150px" }}
                      >
                        Orange
                      </Dropdown.Item>
                    </Dropdown.Menu>
                  </Dropdown>
                </Col>
              </Row>

              <Row>
                <Col md={3}>
                  <Form.Group controlId="neurons">
                    <Form.Label>Number of Neurons in Hidden Layer</Form.Label>
                    <Form.Control
                      type="number"
                      defaultValue={4}
                      placeholder="Enter neurons"
                      onChange={(event) => {
                        setNeurons(eval(event.target.value));
                      }}
                    />
                  </Form.Group>
                </Col>
                <Col md={3}>
                  <Form.Group controlId="activationFunction">
                    <Form.Label>Activation Function Hidden Layer</Form.Label>
                    <Form.Control
                      as="select"
                      onChange={(event) => {
                        setActiationFunction(event.target.value);
                      }}
                    >
                      <option>sigmoid</option>
                      <option>relu</option>
                      <option>tanh</option>
                    </Form.Control>
                  </Form.Group>
                </Col>

                <Col md={3}>
                  <Form.Group controlId="activationFunction">
                    <Form.Label>Activation Function Output Layer</Form.Label>
                    <Form.Control
                      as="select"
                      onChange={(event) => {
                        setActivationFunctionOutputLayer(event.target.value);
                      }}
                    >
                      <option>
                        sigmoid
                        <p style={{ color: "white" }}>-not recomended</p>
                      </option>
                      <option>tanh</option>
                      <option>softmax</option>
                    </Form.Control>
                  </Form.Group>
                </Col>
              </Row>

              <Row>
                <Col md={3}>
                  <Form.Group controlId="learningRate">
                    <Form.Label>Learning Rate</Form.Label>
                    <Form.Control
                      type="number"
                      step="0.1"
                      defaultValue={0.1}
                      min={0}
                      max={1}
                      placeholder="Enter learning rate"
                      onChange={(event) => {
                        setLearningRate(eval(event.target.value));
                      }}
                    />
                  </Form.Group>
                </Col>
                <Col md={3}>
                  <Form.Group controlId="epochs">
                    <Form.Label>Maximum Number of Epochs</Form.Label>
                    <Form.Control
                      defaultValue={10000}
                      type="number"
                      placeholder="Enter epochs"
                      onChange={(event) => {
                        setEpoch(eval(event.target.value));
                      }}
                    />
                  </Form.Group>
                </Col>
              </Row>
              <Col md={3}>
                <Form.Group controlId="goal">
                  <Form.Label>Goal</Form.Label>
                  <Form.Control
                    type="number"
                    defaultValue={0.01}
                    placeholder="Enter goal"
                    onChange={(event) => {
                      setGoal(eval(event.target.value));
                    }}
                  />
                </Form.Group>
              </Col>
              <Row>
                <Col md={3}>
                  <div>Add </div>
                  <Button
                    variant="danger"
                    style={{ width: "70%" }}
                    onClick={addDate}
                  >
                    Add
                  </Button>
                </Col>
                <Col md={3}>
                  <div>Train </div>
                  <Button
                    variant="primary"
                    style={{ width: "70%" }}
                    onClick={trainModel}
                  >
                    Train Model
                  </Button>
                </Col>
                <Col md={3}>
                  <div>Test </div>
                  <Button
                    variant="warning"
                    style={{ width: "70%" }}
                    onClick={() => {
                      testModel();
                    }}
                  >
                    Test Model
                  </Button>
                </Col>
                <Col md={3}>
                  <div>Clear </div>
                  <Button
                    variant="danger"
                    style={{ width: "70%" }}
                    onClick={clear}
                  >
                    Clear
                  </Button>
                </Col>
              </Row>
            </Form>
            <label>Error</label>
            <button style={{ width: "150px", margin: "30px" }}>
              {error.toFixed(7)}
            </button>

            <label>Accuracy</label>
            <button style={{ width: "150px", margin: "30px" }}>
              {accuracy.toFixed(3)}%
            </button>
          </Col>
        </Row>
        <Row>
          {" "}
          <div>Train by File</div>
        </Row>
        <Row>
          <input type="file" id="fileInput" onChange={handleFile} />
        </Row>
      </Container>
    </>
  );
}

export default ConstructGame;
