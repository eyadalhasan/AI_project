// src/App.js
import React from "react";

import ConstructGame from "./constructGame";
import { BrowserRouter, Route } from "react-router-dom";
import { Routes } from "react-router-dom";
import Game from "./Game";
import "./App.css";
import "./index.css";

function App() {
  return (
    <div className="app">
      <ConstructGame />
    </div>
  );
}

export default App;
