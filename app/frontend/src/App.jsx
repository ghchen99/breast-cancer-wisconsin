// src/App.jsx
import React, { useState } from 'react';
import DiagnosisForm from './components/DiagnosisForm';
import PredictionResult from './components/PredictionResult';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handlePrediction = async (formData) => {
    setIsLoading(true);
    setError('');
    setPrediction(null);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      });

      const data = await response.json();

      if (response.ok) {
        setPrediction(data);
      } else {
        setError(data.error || 'Failed to get prediction');
      }
    } catch (err) {
      setError('Failed to connect to the server. Please make sure the backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-4xl mx-auto space-y-8">
        <header className="text-center">
          <h1 className="text-3xl font-bold text-gray-900">
            Breast Cancer Diagnosis Prediction
          </h1>
          <p className="mt-2 text-gray-600">
            Enter cell nuclei measurements to get a prediction
          </p>
        </header>

        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-md text-red-600">
            {error}
          </div>
        )}

        <DiagnosisForm 
          onSubmit={handlePrediction} 
          isLoading={isLoading} 
        />

        {prediction && (
          <PredictionResult prediction={prediction} />
        )}

        <footer className="text-center text-sm text-gray-500 mt-8">
          <p>
            This tool is for educational purposes only. 
            Always consult healthcare professionals for medical decisions.
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;