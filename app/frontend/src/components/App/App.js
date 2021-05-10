import React, { useState } from "react";
import Dropdown from "react-dropdown";
import "react-dropdown/style.css";
import "./App.css";
import SearchIcon from "./search_icon.svg";
import apiService from "../../apis/api";

const options = [
  { value: "elastic", label: "Elasticsearch" },
  { value: "lda", label: "LDA" },
  { value: "doc2vec", label: "Doc2vec" },
  { value: "word2vec", label: "Word2vec" },
  { value: "glove", label: "Glove" },
  { value: "tfidf", label: "Tfidf" },
  { value: "fasttext", label: "Fasttext" },
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
  const [result, setResult] = useState("");

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
    setResult("");
    const cat = category.value;
    // console.log(value, cat);
    if (cat === "elastic") {
      const hits = await apiService.elasticsearch(value);
      const resultArr = await apiService.fetchOriginalSection(hits);
      // console.log(resultArr);
      // let res = "";
      // for (const text of resultArr) {
      //   res += text;
      // }
      const finalText = resultArr.join("\n\n\n--------------\n\n\n");
      console.log(finalText);
      setResult(finalText);
    } else {
      const resultArr = await apiService.basicSearch(value, cat);
      // console.log(resultArr);
      const finalText = resultArr.join("\n\n\n--------------\n\n\n");
      setResult(finalText);
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
      {result && (
        <div className="result">
          <p className="finalText">{result}</p>
        </div>
      )}
    </div>
  );
}

export default App;
