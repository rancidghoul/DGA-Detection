import React, { useState } from "react";

function App() {
  const [domain, setDomain] = useState("");
  const [result, setResult] = useState(null);

  const checkDomain = async () => {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ domain })
    });
    const data = await response.json();
    setResult(data);
  };

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>DGA Detection System</h1>
      <input
        type="text"
        placeholder="Enter domain"
        value={domain}
        onChange={(e) => setDomain(e.target.value)}
        style={{ padding: "8px", marginRight: "10px" }}
      />
      <button onClick={checkDomain} style={{ padding: "8px 15px" }}>
        Check
      </button>
      {result && (
        <div style={{ marginTop: "20px" }}>
          <p><strong>Prediction:</strong> {result.rf_prediction} / {result.lstm_prediction}</p>
          <p><strong>DNS Status:</strong> {result.dns_check}</p>
        </div>
      )}
    </div>
  );
}

export default App;
