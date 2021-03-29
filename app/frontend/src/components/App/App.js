import React, { useState } from "react";
import Dropdown from "react-dropdown";
import "react-dropdown/style.css";
import "./App.css";
import SearchIcon from "./search_icon.svg";

const options = [
  { value: "one", label: "One" },
  { value: "two", label: "Two", className: "myOptionClassName" },
  {
    type: "group",
    name: "group1",
    items: [
      { value: "three", label: "Three", className: "myOptionClassName" },
      { value: "four", label: "Four" },
    ],
  },
  {
    type: "group",
    name: "group2",
    items: [
      { value: "five", label: "Five" },
      { value: "six", label: "Six" },
    ],
  },
];
const defaultOption = options[0];

const onSelect = (some) => {
  console.log(some);
};

function App() {
  const [active, setActive] = useState(false);
  const [value, setValue] = useState("");
  const [error, setError] = useState("");
  const [label, setLabel] = useState("Search query");

  const changeValue = (event) => {
    const value = event.target.value;
    setValue(value);
    setError("");
  };

  const predicted = "";
  const locked = false;
  const fieldClassName = `field ${
    (locked ? active : active || value) && "active"
  } ${locked && !active && "locked"}`;

  return (
    <div className="app">
      <div className="search">
        <Dropdown
          controlClassName="dropdown__main"
          arrowClassName="myArrowClassName"
          options={options}
          onChange={onSelect}
          value={defaultOption}
          placeholder="Select an option"
        />
        <div className={fieldClassName}>
          {active && value && predicted && predicted.includes(value) && (
            <p className="predicted">{predicted}</p>
          )}
          <input
            id={1}
            type="text"
            value={value}
            placeholder={label}
            onChange={changeValue}
            onFocus={() => !locked && setActive(true)}
            onBlur={() => !locked && setActive(false)}
          />
          <label htmlFor={1} className={error && "error"}>
            {error || label}
          </label>
        </div>
        <button className="button">
          <span className="material-icons-outlined">
            <img className="search_icon" src={SearchIcon} alt="search" />
          </span>
        </button>
      </div>
    </div>
  );
}

export default App;
