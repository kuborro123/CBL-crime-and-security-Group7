# CBL-crime-and-security-Group7
# London Crime Data Analysis and Prediction System

## Project Overview

Our project provides comprehensive analysis of crime data in London with a focus on burglary crimes. By combining data visualization and time series forecasting using SARIMAX models, it offers data-driven insights and recommendations for optimizing police resource allocation.

## Key Features

- **Crime Data Analysis**: Statistical analysis of London crime data with emphasis on burglary incidents
- **Data Visualization**: Interactive charts and maps showing crime distribution and trends
- **Time Series Forecasting**: SARIMAX model implementation for predicting future crime patterns
- **Resource Optimization**: Data-driven recommendations for police force deployment

## Tech Stack

- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **Statsmodels**: SARIMAX time series modeling

## Project Structure

```
├── data/                    # Data directory
├── .devcontainer/             # 
├── .idea/                    # 
├── src/                    # documentation
│   └── README.md          # Project documentation
├── visualization/        #visualization outcome graphs
├── Data_loader.py          #data import
├── Dataset_maker.py        #data import
├── requirements.txt        # 
├── streamlit_app.py       #the web to show our allocation
├── Deprivation_prediction.py      #the prediction model
├── time_series_prediction.py       #the primary prediction model
├── Resource_Allocation_repo.py       #polcie allocation file
```

## Installation and Usage

### Installation Steps

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

3. Install dependencies:

### Data Setup


### Running the Analysis

Execute the files in the following order:

1. 

## Module Documentation

### Data Processing Module (`Data_loader.py & Dataset_maker.py`)
- Data cleaning and preprocessing
- Time series data preparation

### Visualization Module (`Data_visualization.py`)
- Crime heatmap generation
- Temporal trend analysis
- Geographic distribution mapping

### Modeling Module (`Deprivation_prediction.py`)
- SARIMAX model construction
- Parameter optimization
- Forecast generation and validation

### Optimization Module (`Resource_Allocation_repo.py`)
- Prediction-based resource allocation algorithms
- Police deployment optimization

## Output and Results

- **Visualization Reports**: Charts, maps, and interactive dashboards
- **Prediction Results**: Future crime trend forecasts
- **Optimization Recommendations**: Police resource allocation strategies
- **Analysis Reports**: Comprehensive project documentation

## Data Sources

- London Metropolitan Police Crime Data
- Additional demographic and socioeconomic datasets


## Contact

- Project Maintainer: Group 7
- Email: 
- Project Link: https://github.com/kuborro123/CBL-crime-and-security-Group7


---

**Disclaimer**: This project is for academic and educational purposes only. 