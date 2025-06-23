# London Crime Data Analysis and Resource Allocation

## Project Overview

This project analyzes crime data in London with a focus on burglary crimes. It uses data visualization and time series forecasting with SARIMAX models to provide insights and recommendations for police resource allocation.

## Features

- Crime data analysis focusing on burglary incidents in London
- Data visualization with charts and maps showing crime patterns
- Time series forecasting using SARIMAX models
- Police resource allocation recommendations
- Interactive web application built with Streamlit

## Technology Used

- Python
- Pandas for data processing
- Matplotlib and Seaborn for visualization
- Statsmodels for SARIMAX modeling
- Streamlit for web application
- Plotly for interactive charts

## Project Structure

```
CBL-crime-and-security-Group7/
├── data/                           # Data files
├── src/                           # Source code
├── visualization/                  # Generated charts and graphs
├── Data_loader.py                 # Data loading functions
├── Dataset_maker.py               # Data preprocessing
├── requirements.txt               # Required packages
├── streamlit_app.py              # Web application
├── Deprivation_prediction.py     # Deprivation prediction model
├── time_series_prediction.py     # Time series forecasting
└── Resource_Allocation_repo.py   # Resource allocation algorithms
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kuborro123/CBL-crime-and-security-Group7
   cd CBL-crime-and-security-Group7
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. Run the data processing scripts first:
   ```bash
   python Data_loader.py
   python Dataset_maker.py
   ```

2. Run the prediction models:
   ```bash
   python time_series_prediction.py
   python Deprivation_prediction.py
   ```

3. Generate resource allocation recommendations:
   ```bash
   python Resource_Allocation_repo.py
   ```

4. Launch the web application:
   ```bash
   streamlit run streamlit_app.py
   ```

The web app will open at http://localhost:8501

## What the Project Does

### Data Analysis
- Processes London crime data
- Focuses on burglary crime patterns
- Analyzes trends over time and across different areas

### Visualization
- Creates crime heat maps
- Shows temporal trends in crime data
- Maps crime distribution across London boroughs

### Prediction
- Uses SARIMAX models for time series forecasting
- Predicts future crime patterns
- Considers seasonal and trend components

### Resource Allocation
- Recommends optimal police deployment
- Based on predicted crime patterns
- Aims to improve resource efficiency

## Output

The project generates:
- Interactive charts and maps
- Crime trend forecasts
- Police resource allocation recommendations
- Summary reports

## Data Sources

- London Metropolitan Police crime data
- Demographic and socioeconomic data
- Geographic data for London areas

## Group Information

This project was developed by Group 7 as part of a university course.

Repository: https://github.com/kuborro123/CBL-crime-and-security-Group7

## Note

This project is for educational purposes only. The analysis and recommendations are part of academic coursework and should not be used for actual police operations.
```
