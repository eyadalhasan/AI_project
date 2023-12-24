// src/App.js
import React, { useEffect, useRef } from "react";
import { Container, Row, Col, Form, Button } from "react-bootstrap";
import { useState } from "react";
import Banana from "./banana.png";
import Orange from "./Orange.png";
import Apple from "./apple (1).png";

import { Dropdown } from "react-bootstrap"; // Import Bootstrap Dropdown component
import Swal from "sweetalert2";
function ConstructGame() {
  const [images, setImages] = useState([]);
  const [selectedOption, setSelectedOption] = useState("Apple");
  const [sweetness, setSweetness] = useState(0);
  const [colorValue, setColorValue] = useState(0);
  const [fruitValue, setFruitValue] = useState([1, 0, 0]);
  const [dataSet, setDataSet] = useState([]);
  const [normalizedData, setNormalizedDate] = useState([]);
  const [neurons, setNeurons] = useState(1);
  const [activationFunction, setActiationFunction] = useState("relu");
  const [learningRate, setLearningRate] = useState(0);
  const [epoch, setEpoch] = useState(0);
  const [goal, setGoal] = useState(0);

  const handleSelect = (eventKey, event) => {
    // Update the selected option when an option is clicked
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
  // useEffect(() => {
  //   if (neurons != 0) {
  //     network();
  //   }
  // }, [normalizedData]);
  function drawLine(m, b) {
    const canvas = document.getElementById("myCanvas");
    const ctx = canvas.getContext("2d");
    const startX = 0;
    const endX = canvas.width;

    // Calculate corresponding y values
    const startY = m * startX + b;
    const endY = m * endX + b;

    // Draw the line
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(endX, endY);
    ctx.strokeStyle = "blue";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

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
  function relu(x) {
    return Math.max(0.01 * x, x);
  }
  const getRandomNumber = (min, max) => Math.random() * (max - min) + min;
  function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }
  function tanh(x) {
    return Math.tanh(x);
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

  function sigmoidDerivative(x) {
    const sig = sigmoid(x);
    return sig * (1 - sig);
  }
  function tanhDerivative(x) {
    const tanhx = tanh(x);
    return 1 - tanhx * tanhx;
  }
  function reluDerivative(x) {
    return x < 0 ? 0.01 : 1;
  }
  function calcualteError(deNormalizedoutput, normalizedData) {
    var error = [];
    var gradienterror = 0;
    error = deNormalizedoutput.map((item, key) => {
      let a = -(deNormalizedoutput[key] - normalizedData["fruit"][key]);
      return a;
    });
    // console.log(error);
    return error;
  }
  var inputWights = useRef([]);
  var hiddenLayer = useRef([]);
  var outputLayerthreshold = useRef([]);
  function network() {
    var gradienterror = [];
    // inputWights.current = [];
    // hiddenLayer.current = [];
    // outputLayerthreshold.current = [];
    var hiddenOutput = [];
    var actualOutput = [];
    var inputVector = [];
    var deltW = 0;
    var nodeweight = [];
    var deltaTh = 0;
    var hiddenerror = 0;
    var gradienterrorHidden = [];
    var inputX = [];
    var er = 0;
    var epochiscounter = epoch;

    if (neurons == 0) {
      alert("You Have to enter number of  neurons in hidden layer");
    } else {
      for (let i = 0; i < 2; i++) {
        nodeweight = [];
        for (let j = 0; j < neurons; j++) {
          nodeweight.push(getRandomNumber(minRange, maxRange));
        }
        inputWights.current.push(nodeweight);
      }
      for (let i = 0; i < neurons; i++) {
        hiddenLayer.current.push({
          weight: [
            getRandomNumber(minRange, maxRange),
            getRandomNumber(minRange, maxRange),
            getRandomNumber(minRange, maxRange),
          ],
          threshold: getRandomNumber(minRange, maxRange),
        });
      }
    }
    for (let i = 0; i < 3; i++) {
      outputLayerthreshold.current.push(getRandomNumber(minRange, maxRange));
    }
    // ******************************************
    while (epochiscounter > 0) {
      epochiscounter--;

      // for (let i = 0; i < normalizedData.length; i++) {
      //   var x = 0;
      //   var yh = 0;
      //   hiddenOutput = [];

      //   for (let j = 0; j < neurons && normalizedData.length != 0; j++) {
      //     console.log("data", normalizedData[i]);
      //     x =
      //       normalizedData[i]["sweetness"] * inputWights.current[0][j] +
      //       normalizedData[i]["color"] * inputWights.current[1][j] +
      //       hiddenLayer.current[j]["threshold"];
      //     inputX.push(x);

      //     if (activationFunction == "relu") {
      //       yh = relu(x);
      //       hiddenOutput.push(yh);
      //     } else if (activationFunction == "sigmoid") {
      //       yh = sigmoid(x);
      //       hiddenOutput.push(yh);
      //     } else if (activationFunction == "tanh") {
      //       yh = tanh(x);
      //       hiddenOutput.push(yh);
      //     }
      //   }
      //   if (normalizedData.length != 0) {
      //     inputVector = [];
      //     for (let j = 0; j < 3; j++) {
      //       let z = 0;
      //       for (let k = 0; k < neurons; k++) {
      //         z = z + hiddenOutput[k] * hiddenLayer.current[k]["weight"][j];
      //       }
      //       z = z + outputLayerthreshold.current[j];
      //       inputVector.push(z);
      //     }
      //     // console.log("before softmax", inputVector);

      //     let actualOutput = softmax(inputVector);
      //     // console.log("result", actualOutput);
      //     var deNormalizedoutput = actualOutput.map((output, i) => {
      //       return deNormalize(output, 0, 1);
      //     });

      //     var error = calcualteError(deNormalizedoutput, normalizedData[i]);
      //     // console.log(error);

      //     //UPDATE WIEGHTS FOR HIDDEN LAYER
      //     for (let a = 0; a < 3; a++) {
      //       gradienterror.push(softmaxDerivative(inputVector, a) * error[a]);
      //       deltW = learningRate * gradienterror[a] * inputVector[a];
      //       deltaTh = learningRate * -1 * gradienterror[a];

      //       outputLayerthreshold.current[a] += deltaTh;
      //       if (outputLayerthreshold.current[a] > 1.2) {
      //         outputLayerthreshold.current[a] = 1.2;
      //       } else if (outputLayerthreshold.current[a] < -1.2) {
      //         outputLayerthreshold.current[a] = -1.2;
      //       }

      //       for (let m = 0; m < neurons; m++) {
      //         hiddenLayer.current[m]["weight"][a] += deltW;
      //         if (hiddenLayer.current[m]["weight"][a] > 1.2) {
      //           hiddenLayer.current[m]["weight"][a] = 1.2;
      //         }
      //         if (hiddenLayer.current[m]["weight"][a] < -1.2) {
      //           hiddenLayer.current[m]["weight"][a] = -1.2;
      //         }
      //       }
      //     }

      //     // console.log("hidden after", hiddenLayer);

      //     for (let a = 0; a < neurons; a++) {
      //       hiddenerror = 0;
      //       for (let m = 0; m < 3; m++) {
      //         hiddenerror +=
      //           hiddenLayer.current[a]["weight"][m] * gradienterror[m];
      //       }
      //       // console.log("hidden", hiddenerror);

      //       if (activationFunction == "sigmoid") {
      //         er = hiddenerror * sigmoidDerivative(inputX[a]);
      //         gradienterrorHidden.push(er);
      //       } else if (activationFunction == "tanh") {
      //         er = hiddenerror * tanhDerivative(inputX[a]);
      //         // console.log("", er);
      //         gradienterrorHidden.push(er);
      //       }

      //       if (activationFunction == "relu") {
      //         // console.log("hidden error", hiddenerror);
      //         er = hiddenerror * reluDerivative(inputX[a]);

      //         gradienterrorHidden.push(er);
      //       }
      //       deltaTh = learningRate * -1 * gradienterrorHidden[a];
      //       hiddenLayer.current[a].threshold += deltaTh;
      //       if (hiddenLayer.current[a].threshold > 1.2) {
      //         hiddenLayer.current[a].threshold = 1.2;
      //       } else if (hiddenLayer.current[a].threshold < -1.2) {
      //         hiddenLayer.current[a].threshold = -1.2;
      //       }

      //       // Corrected update for weights
      //       for (let o = 0; o < 2; o++) {
      //         deltW = learningRate * gradienterrorHidden[a] * inputX[a];
      //         inputWights.current[o][a] += deltW;
      //         if (inputWights.current[o][a] > 1.2) {
      //           inputWights.current[o][a] = 1.2;
      //         }
      //         if (inputWights.current[o][a] < -1.2) {
      //           inputWights.current[o][a] = -1.2;
      //         }
      //       }

      //       // console.log("input after", inputWights.current);
      //     }
      //   }
      // }

      
    }
  }
  

  function testModel() {
    if (neurons === 0) {
      alert("You need to train the model first");
      return;
    }

    const inputSweetness = sweetness;
    console.log(inputSweetness);
    const inputColor = colorValue;
    console.log(inputColor);

    // Normalize the test input
    const normalizedInput = {
      sweetness: normalize(inputSweetness, 0, 1),
      color: normalize(inputColor, 1, 3),
    };
    console.log(normalizedInput);

    // Feedforward through the trained network
    let hiddenOutput = [];
    for (let j = 0; j < neurons; j++) {
      const x =
        normalizedInput.sweetness * inputWights.current[0][j] +
        normalizedInput.color * inputWights.current[1][j] -
        hiddenLayer.current[j].threshold;

      if (activationFunction === "relu") {

        
        hiddenOutput.push(relu(x));
      } else if (activationFunction === "sigmoid") {
        hiddenOutput.push(sigmoid(x));
      } else if (activationFunction === "tanh") {
        hiddenOutput.push(tanh(x));
      }
    }

    let inputVector = [];
    for (let j = 0; j < 3; j++) {
      let z = 0;
      for (let k = 0; k < neurons; k++) {
        z += hiddenOutput[k] * hiddenLayer.current[k].weight[j];
      }
      z -= outputLayerthreshold.current[j];
      inputVector.push(z);
    }
    console.log("TESTING", inputVector);

    const actualOutput = softmax(inputVector);
    console.log(actualOutput);

    // Denormalize the output
    const deNormalizedOutput = actualOutput.map((output, i) => {
      return deNormalize(output, 0, 1);
    });

    // Display the result
    console.log("Test Result:", deNormalizedOutput);
    const max = Math.max(...deNormalizedOutput);
    const maxIndex = deNormalizedOutput.indexOf(max);
    var title = "";
    var imageurl = "";
    if (maxIndex == 0) {
      title = "Apple";
      imageurl = Apple;
    } else if (maxIndex == 1) {
      title = "Banana";
      imageurl = Banana;
    } else if (maxIndex == 2) {
      title = "Orange";
      imageurl = Orange;
    }

    Swal.fire({
      title: title,

      imageUrl: imageurl, // Replace with the path to your image
      imageAlt: "Custom Image Alt Text", // Replace with the alt text for your image
      showCloseButton: true,
      confirmButtonText: "Close",
    });

    // You can use the output as needed in your application
  }

  function trainModel() {
    network();
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
                      placeholder="Enter neurons"
                      onChange={(event) => {
                        setNeurons(eval(event.target.value));
                      }}
                    />
                  </Form.Group>
                </Col>
                <Col md={3}>
                  <Form.Group controlId="activationFunction">
                    <Form.Label>Activation Function</Form.Label>
                    <Form.Control
                      as="select"
                      onChange={(event) => {
                        setActiationFunction(event.target.value);
                      }}
                    >
                      <option>relu</option>
                      <option>sigmoid</option>
                      <option>tanh</option>
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
                    onChange={clear}
                  >
                    Clear
                  </Button>
                </Col>
              </Row>
            </Form>
          </Col>
        </Row>
      </Container>
    </>
  );
}

export default ConstructGame;
