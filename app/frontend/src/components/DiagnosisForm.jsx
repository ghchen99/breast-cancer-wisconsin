// src/components/DiagnosisForm.jsx
import React, { useState } from 'react';
import { Loader2 } from 'lucide-react';

const DiagnosisForm = ({ onSubmit, isLoading }) => {
  const [formData, setFormData] = useState({
    radius_mean: '', texture_mean: '', perimeter_mean: '', area_mean: '', 
    smoothness_mean: '', compactness_mean: '', concavity_mean: '', 
    'concave points_mean': '', symmetry_mean: '', fractal_dimension_mean: '',
    radius_se: '', texture_se: '', perimeter_se: '', area_se: '', 
    smoothness_se: '', compactness_se: '', concavity_se: '', 
    'concave points_se': '', symmetry_se: '', fractal_dimension_se: '',
    radius_worst: '', texture_worst: '', perimeter_worst: '', area_worst: '', 
    smoothness_worst: '', compactness_worst: '', concavity_worst: '', 
    'concave points_worst': '', symmetry_worst: '', fractal_dimension_worst: ''
  });

  const [error, setError] = useState('');

  // Validation ranges for each measurement type
  const validationRanges = {
    radius: { min: 6.0, max: 30.0 },
    texture: { min: 9.0, max: 40.0 },
    perimeter: { min: 40.0, max: 190.0 },
    area: { min: 140.0, max: 2600.0 },
    smoothness: { min: 0.05, max: 0.16 },
    compactness: { min: 0.02, max: 0.35 },
    concavity: { min: 0.0, max: 0.5 },
    'concave points': { min: 0.0, max: 0.2 },
    symmetry: { min: 0.1, max: 0.3 },
    'fractal_dimension': { min: 0.05, max: 0.1 }
  };

  // Get description for each field
  const getFieldDescription = (field) => {
    const base = field.split('_')[0].replace('concave points', 'concave_points');
    
    const descriptions = {
      radius: 'Mean of distances from center to points on the perimeter',
      texture: 'Standard deviation of gray-scale values',
      perimeter: 'Perimeter of the cell nucleus',
      area: 'Area of the cell nucleus',
      smoothness: 'Local variation in radius lengths',
      compactness: 'Calculated as (perimeter² / area - 1.0)',
      concavity: 'Severity of concave portions of the contour',
      'concave_points': 'Number of concave portions of the contour',
      symmetry: 'Symmetry of the cell nucleus',
      'fractal_dimension': '"Coastline approximation" - 1'
    };

    return descriptions[base] || '';
  };

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

    // Convert all values to numbers
    const numericData = {};
    for (const [key, value] of Object.entries(formData)) {
      const num = parseFloat(value);
      if (isNaN(num)) {
        setError('Please enter valid numbers');
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
    'Mean Values': Object.keys(formData).filter(k => k.includes('_mean')),
    'Standard Error': Object.keys(formData).filter(k => k.includes('_se')),
    'Worst Values': Object.keys(formData).filter(k => k.includes('_worst'))
  };

  return (
    <div className="w-full max-w-4xl mx-auto bg-white rounded-lg shadow p-6">
      <h2 className="text-2xl font-bold mb-6">Enter Cell Measurements</h2>
      
      <form onSubmit={handleSubmit} className="space-y-6">
        {Object.entries(fieldGroups).map(([groupName, fields]) => (
          <div key={groupName} className="space-y-4">
            <h3 className="text-lg font-semibold">{groupName}</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {fields.map(field => {
                const baseFeature = field.split('_')[0];
                const range = validationRanges[baseFeature] || validationRanges['concave points'];
                
                return (
                  <div key={field} className="space-y-2">
                    <label className="block text-sm font-medium text-gray-700">
                      {baseFeature.charAt(0).toUpperCase() + baseFeature.slice(1)}
                      <span className="block text-xs text-gray-500" title={getFieldDescription(field)}>
                        {getFieldDescription(field)}
                      </span>
                      {range && (
                        <span className="block text-xs text-gray-400">
                          Range: {range.min} - {range.max}
                        </span>
                      )}
                    </label>
                    <input
                      type="number"
                      name={field}
                      value={formData[field]}
                      onChange={handleInputChange}
                      step="any"
                      min={range?.min}
                      max={range?.max}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
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
            className={`px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 
              focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
              disabled:opacity-50 disabled:cursor-not-allowed flex items-center`}
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <Loader2 className="animate-spin mr-2" />
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