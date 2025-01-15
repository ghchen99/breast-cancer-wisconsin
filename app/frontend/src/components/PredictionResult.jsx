// src/components/PredictionResult.jsx
import React from 'react';
import { CheckCircle, AlertCircle, AlertTriangle } from 'lucide-react';

const PredictionResult = ({ prediction }) => {
  if (!prediction) return null;

  const { diagnosis, confidence, confidence_level, malignant_probability, benign_probability, warning } = prediction;
  const isBenign = diagnosis === 'Benign';

  const getConfidenceColor = (level) => {
    switch(level) {
      case 'Very High': return 'text-green-600';
      case 'High': return 'text-blue-600';
      case 'Moderate': return 'text-yellow-600';
      case 'Low': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto mt-6 bg-white rounded-lg shadow p-6">
      <div className="flex items-center gap-2 mb-4">
        {isBenign ? (
          <CheckCircle className="text-green-500 h-6 w-6" />
        ) : (
          <AlertCircle className="text-red-500 h-6 w-6" />
        )}
        <h2 className="text-xl font-bold">Diagnosis Prediction Result</h2>
      </div>

      <div className="space-y-4">
        {/* Main Prediction Alert */}
        <div className={`p-4 rounded-md ${isBenign ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'}`}>
          <div className="text-lg">
            <div className="font-semibold">Predicted Diagnosis: {diagnosis}</div>
            <div className="text-sm mt-1">
              Confidence Level: 
              <span className={`font-medium ${getConfidenceColor(confidence_level)}`}>
                {' '}{confidence_level} ({(confidence * 100).toFixed(1)}%)
              </span>
            </div>
          </div>
        </div>

        {/* Warning for low confidence if present */}
        {warning && (
          <div className="flex gap-2 p-4 bg-yellow-50 border border-yellow-200 rounded-md">
            <AlertTriangle className="h-5 w-5 text-yellow-600 shrink-0" />
            <div>
              <div className="font-medium text-yellow-600">Warning</div>
              <div className="text-yellow-700">{warning}</div>
            </div>
          </div>
        )}

        {/* Detailed Probabilities */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium">Detailed Probabilities:</h3>
          
          {/* Probability Bars */}
          <div className="space-y-3">
            {/* Benign Probability */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Benign</span>
                <span>{(benign_probability * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 bg-gray-200 rounded">
                <div 
                  className="h-full bg-green-500 rounded"
                  style={{ width: `${benign_probability * 100}%` }}
                />
              </div>
            </div>
            
            {/* Malignant Probability */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>Malignant</span>
                <span>{(malignant_probability * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 bg-gray-200 rounded">
                <div 
                  className="h-full bg-red-500 rounded"
                  style={{ width: `${malignant_probability * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Confidence Level Note */}
        <div className="text-sm text-gray-600 mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
          <p className="font-medium mb-2">Understanding the Confidence Level:</p>
          <ul className="space-y-1 list-disc pl-4">
            <li><span className="text-green-600 font-medium">Very High</span>: The model is extremely certain of its prediction (≥90% confidence)</li>
            <li><span className="text-blue-600 font-medium">High</span>: The model is quite confident (80-89% confidence)</li>
            <li><span className="text-yellow-600 font-medium">Moderate</span>: The model is reasonably confident (70-79% confidence)</li>
            <li><span className="text-red-600 font-medium">Low</span>: The model shows significant uncertainty (&lt;70% confidence)</li>
          </ul>
          <p className="mt-2 italic">
            Note: All predictions should be confirmed by healthcare professionals regardless of confidence level.
          </p>
        </div>
      </div>
    </div>
  );
};

export default PredictionResult;