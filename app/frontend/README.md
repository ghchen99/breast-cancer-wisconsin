# Breast Cancer Diagnosis Frontend

This guide explains how to build the React frontend for the Breast Cancer Diagnosis application from scratch. The frontend provides a user interface for entering cell measurements and displays prediction results from our machine learning model.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Project Setup](#project-setup)
- [Installing Dependencies](#installing-dependencies)
- [Project Structure](#project-structure)
- [Component Development](#component-development)
- [Styling with Tailwind](#styling-with-tailwind)
- [State Management](#state-management)
- [API Integration](#api-integration)
- [Running the Application](#running-the-application)

## Prerequisites

- Node.js (v18+ recommended)
- npm (comes with Node.js)
- Basic knowledge of React and JavaScript
- Backend server running on port 5000 (see backend README)

## Project Setup

### 1. Initial Vite Setup

First, create a new Vite project with React. Run the following command and follow the prompts:
```bash
npm create vite@latest
```

When prompted, select:
- Project name: `frontend`
- Framework: `React`
- Variant: `JavaScript`

This command creates the following structure:
```
frontend/
├── public/               # Static assets directory
│   └── vite.svg         # Vite logo asset
├── src/                 # Source code directory
│   ├── assets/          # Project assets
│   │   └── react.svg    # React logo asset
│   ├── App.css          # Default App styles
│   ├── App.jsx          # Main App component
│   ├── index.css        # Global styles
│   └── main.jsx         # Entry point
├── .gitignore           # Git ignore configuration
├── package.json         # Project configuration
└── vite.config.js       # Vite configuration
```

### 2. Installing Dependencies

Navigate to the project directory and install dependencies:
```bash
cd frontend
npm install
```

This creates:
- `node_modules/` directory containing all installed packages
- `package-lock.json` file that locks dependency versions

### 3. Setting up Tailwind CSS

Install and initialize Tailwind:
```bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

This creates:
- `postcss.config.js` - PostCSS configuration for Tailwind
- `tailwind.config.js` - Tailwind configuration file

### 4. Setting up ESLint

Install ESLint and its plugins:
```bash
npm install -D eslint @eslint/js eslint-plugin-react eslint-plugin-react-hooks eslint-plugin-react-refresh
```

Initialize ESLint configuration:
```bash
npm init @eslint/config
```

This creates:
- `eslint.config.js` - ESLint configuration file

After all these steps, your project structure should look like this:
```
frontend/
├── node_modules/          # Created by npm install
├── public/               # Created by Vite
│   └── vite.svg         # Default Vite asset
├── src/                 # Created by Vite
│   ├── assets/          # Created by Vite
│   │   └── react.svg    # Default React asset
│   ├── App.css          # Created by Vite
│   ├── App.jsx          # Created by Vite
│   ├── index.css        # Created by Vite
│   └── main.jsx         # Created by Vite
├── .gitignore           # Created by Vite
├── package.json         # Created by Vite
├── package-lock.json    # Created by npm install
├── eslint.config.js     # Created by ESLint setup
├── postcss.config.js    # Created by Tailwind setup
├── tailwind.config.js   # Created by Tailwind setup
└── vite.config.js       # Created by Vite
```

## Component Development

### 1. Main App Component (App.jsx)
The App component serves as the main container and handles:
- State management for predictions
- Error handling
- API communication
- Layout structure

### 2. DiagnosisForm Component
Features:
- Input fields for 30 different measurements
- Field validation with proper ranges
- Loading state handling
- Error messages
- Reset functionality

### 3. PredictionResult Component
Features:
- Displays prediction outcome (Benign/Malignant)
- Shows confidence levels with color coding
- Probability visualization with progress bars
- Warning messages for low confidence predictions

## Styling with Tailwind

1. Configure Tailwind (tailwind.config.js):
```javascript
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

2. Set up global styles (index.css):
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  font-family: Inter, system-ui, Avenir, Helvetica, Arial, sans-serif;
}
```

## State Management

The application uses React's built-in useState hook for state management:

- Form data state in DiagnosisForm
- Prediction results in App component
- Loading states
- Error handling states

## API Integration

The frontend communicates with the Flask backend using the Fetch API:

```javascript
const response = await fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(formData)
});
```

## Running the Application

1. Start the development server:
```bash
npm run dev
```

2. Build for production:
```bash
npm run build
```

## Development Guidelines

1. Component Organization
- Keep components focused and single-responsibility
- Use proper prop types
- Implement error boundaries
- Handle loading states

2. Styling Practices
- Use Tailwind utility classes
- Maintain consistent spacing
- Ensure responsive design
- Follow accessibility guidelines

3. Form Handling
- Implement proper validation
- Show clear error messages
- Provide user feedback
- Handle edge cases

4. Error Handling
- Graceful error presentation
- Clear error messages
- Network error handling
- Validation error handling

## Additional Features

1. Form Validation
- Range checking for all measurements
- Required field validation
- Numerical input validation

2. User Experience
- Loading indicators
- Clear error messages
- Responsive design
- Reset functionality

3. Accessibility
- ARIA labels
- Keyboard navigation
- Focus management
- Screen reader support

## Production Considerations

1. Performance
- Code splitting
- Lazy loading
- Image optimization
- Bundle size optimization

2. Security
- Input sanitization
- CORS configuration
- Error handling
- Data validation

3. Maintenance
- Clear documentation
- Code comments
- Consistent formatting
- Version control

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details