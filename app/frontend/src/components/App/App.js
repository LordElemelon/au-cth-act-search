import React, { useState } from "react";
import Dropdown from "react-dropdown";
import "react-dropdown/style.css";
import "./App.css";
import SearchIcon from "./search_icon.svg";
import apiService from "../../apis/api";

const options = [
  { value: "elastic", label: "Elasticsearch" },
  { value: "word2vec", label: "Word2vec" },
  { value: "doc2vec", label: "Doc2vec" },
  { value: "fasttext", label: "Fasttext" },
  { value: "glove", label: "Glove" },
  { value: "tfidf", label: "Tfidf" },
];

function App() {
  const [active, setActive] = useState(false);
  const [value, setValue] = useState("");
  const [error, setError] = useState("");
  const [label] = useState("Keywords");
  const [category, setCategory] = useState({
    value: "elastic",
    label: "Elasticsearch",
  });
  const [result, setResult] = useState("HAHSDHSHAD");

  const onSelect = (some) => {
    setCategory(some);
  };

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

  const onSearch = async () => {
    if (!value) {
      setResult("");
      return;
    }
    const cat = category.value;
    console.log(value, cat);
    if (cat === "elastic") {
      const hits = await apiService.elasticsearch(value);
      const resultArr = await apiService.fetchOriginalSection(hits);
      console.log(resultArr);
      // let res = "";
      // for (const text of resultArr) {
      //   res += text;
      // }
      setResult(resultArr.join("\n"));
    } else {
      const resultArr = await apiService.basicSearch(value, cat);
      console.log(resultArr);
      setResult(resultArr.join("\n"));
    }
  };

  return (
    <div className="app">
      <div className="search">
        <Dropdown
          controlClassName="dropdown__main"
          arrowClassName="myArrowClassName"
          options={options}
          onChange={onSelect}
          value={category}
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
        <button className="button" onClick={onSearch}>
          <span className="material-icons-outlined">
            <img className="search_icon" src={SearchIcon} alt="search" />
          </span>
        </button>
      </div>
      {result && <div className="result">{result}</div>}
    </div>
  );
}

export default App;
