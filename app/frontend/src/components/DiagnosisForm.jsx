import React, { useState } from 'react';
import { Loader2 } from 'lucide-react';

const DiagnosisForm = ({ onSubmit, isLoading }) => {
  // Validation ranges matched with backend
  const validationRanges = {
    // Mean values
    'radius_mean': { min: 10.07, max: 22.51, description: 'Mean of distances from center to points on the perimeter' },
    'texture_mean': { min: 15.26, max: 24.21, description: 'Standard deviation of gray-scale values' },
    'perimeter_mean': { min: 65.13, max: 149.65, description: 'Mean perimeter of the cell nucleus' },
    'area_mean': { min: 233.14, max: 1484.66, description: 'Mean area of the cell nucleus' },
    'smoothness_mean': { min: 0.069, max: 0.124, description: 'Mean local variation in radius lengths' },
    'compactness_mean': { min: 0.01, max: 0.276, description: 'Mean of perimeter² / area - 1.0' },
    'concavity_mean': { min: 0.01, max: 0.351, description: 'Mean severity of concave portions of contour' },
    'concave points_mean': { min: 0.01, max: 0.158, description: 'Mean number of concave portions of contour' },
    'symmetry_mean': { min: 0.128, max: 0.222, description: 'Mean symmetry of the cell nucleus' },
    'fractal_dimension_mean': { min: 0.051, max: 0.076, description: 'Mean "coastline approximation" - 1' },
    
    // Standard Error values
    'radius_se': { min: 0.15, max: 1.012, description: 'Standard error of distances from center to perimeter' },
    'texture_se': { min: 0.289, max: 2.310, description: 'Standard error of gray-scale values' },
    'perimeter_se': { min: 0.735, max: 5.360, description: 'Standard error of perimeter' },
    'area_se': { min: 10.0, max: 130.44, description: 'Standard error of area' },
    'smoothness_se': { min: 0.0039, max: 0.0114, description: 'Standard error of smoothness' },
    'compactness_se': { min: 0.01, max: 0.0772, description: 'Standard error of compactness' },
    'concavity_se': { min: 0.001, max: 0.1014, description: 'Standard error of concavity' },
    'concave points_se': { min: 0.0006, max: 0.0274, description: 'Standard error of concave points' },
    'symmetry_se': { min: 0.0085, max: 0.0282, description: 'Standard error of symmetry' },
    'fractal_dimension_se': { min: 0.001, max: 0.0075, description: 'Standard error of fractal dimension' },
    
    // Worst values (complete set matching backend)
    'radius_worst': { min: 9.95, max: 25.59, description: 'Worst radius (largest mean value for radius)' },
    'texture_worst': { min: 17.42, max: 34.05, description: 'Worst texture (largest mean value for texture)' },
    'perimeter_worst': { min: 71.73, max: 163.14, description: 'Worst perimeter (largest mean value for perimeter)' },
    'area_worst': { min: 189.64, max: 1929.16, description: 'Worst area (largest mean value for area)' },
    'smoothness_worst': { min: 0.089, max: 0.184, description: 'Worst smoothness (largest mean value for smoothness)' },
    'compactness_worst': { min: 0.01, max: 0.893, description: 'Worst compactness (largest mean value for compactness)' },
    'concavity_worst': { min: 0.01, max: 1.162, description: 'Worst concavity (largest mean value for concavity)' },
    'concave points_worst': { min: 0.01, max: 0.311, description: 'Worst concave points (largest mean value for concave points)' },
    'symmetry_worst': { min: 0.182, max: 0.366, description: 'Worst symmetry (largest mean value for symmetry)' },
    'fractal_dimension_worst': { min: 0.044, max: 0.132, description: 'Worst fractal dimension (largest mean value for fractal dimension)' }
  };

  const [formData, setFormData] = useState(
    Object.keys(validationRanges).reduce((acc, key) => ({ ...acc, [key]: '' }), {})
  );

  const [error, setError] = useState('');

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    if (error) setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    // Validate all fields are filled
    if (Object.values(formData).some(v => v === '')) {
      setError('Please fill in all fields');
      return;
    }

    // Convert and validate all values
    const numericData = {};
    for (const [key, value] of Object.entries(formData)) {
      const num = parseFloat(value);
      if (isNaN(num)) {
        setError(`Invalid number for ${key}`);
        return;
      }
      
      const range = validationRanges[key];
      if (num < range.min || num > range.max) {
        setError(`${key} must be between ${range.min} and ${range.max}`);
        return;
      }
      
      numericData[key] = num;
    }

    onSubmit(numericData);
  };

  const handleReset = () => {
    setFormData(Object.fromEntries(
      Object.keys(formData).map(key => [key, ''])
    ));
    setError('');
  };

  // Group fields by measurement type
  const fieldGroups = {
    'Mean Values': Object.keys(validationRanges).filter(k => k.includes('_mean')),
    'Standard Error': Object.keys(validationRanges).filter(k => k.includes('_se')),
    'Worst Values': Object.keys(validationRanges).filter(k => k.includes('_worst'))
  };

  const formatRange = (min, max) => {
    const formatNumber = (num) => {
      if (num < 0.01) return num.toExponential(2);
      if (num < 1) return num.toFixed(3);
      if (Number.isInteger(num)) return num.toString();
      return num.toFixed(2);
    };
    return `${formatNumber(min)} - ${formatNumber(max)}`;
  };

  return (
    <div className="w-full max-w-4xl mx-auto bg-white rounded-lg shadow p-6">
      <h2 className="text-2xl font-bold mb-6">Enter Cell Measurements</h2>
      
      <form onSubmit={handleSubmit} className="space-y-6">
        {Object.entries(fieldGroups).map(([groupName, fields]) => (
          <div key={groupName} className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-800 border-b pb-2">{groupName}</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {fields.map(field => {
                const range = validationRanges[field];
                return (
                  <div key={field} className="space-y-2">
                    <label className="block">
                      <span className="text-sm font-medium text-gray-700">
                        {field.split('_')[0].charAt(0).toUpperCase() + field.split('_')[0].slice(1)}
                      </span>
                      <span className="block text-xs text-gray-500" title={range.description}>
                        {range.description}
                      </span>
                      <span className="block text-xs text-gray-400">
                        Range: {formatRange(range.min, range.max)}
                      </span>
                    </label>
                    <input
                      type="number"
                      name={field}
                      value={formData[field]}
                      onChange={handleInputChange}
                      step="any"
                      min={range.min}
                      max={range.max}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                      placeholder={`${range.min} - ${range.max}`}
                    />
                  </div>
                );
              })}
            </div>
          </div>
        ))}
        
        <div className="flex gap-4">
          <button 
            type="submit" 
            className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 
              focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
              disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <Loader2 className="animate-spin mr-2 h-4 w-4" />
                Processing...
              </>
            ) : (
              'Predict'
            )}
          </button>
          <button 
            type="button" 
            onClick={handleReset}
            className="px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50"
            disabled={isLoading}
          >
            Reset
          </button>
        </div>

        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-md text-red-600">
            {error}
          </div>
        )}
      </form>
    </div>
  );
};

export default DiagnosisForm;