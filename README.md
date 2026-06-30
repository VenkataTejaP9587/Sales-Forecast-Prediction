# Sales Forecast Prediction

This project is a Streamlit-based sales forecasting application that predicts sales trends for products using machine learning. It includes a login/signup flow, data upload, model training, forecasting, and interactive visualizations.

## Project Overview

The application helps users:
- upload a sales dataset
- train a regression model to predict sales
- view item-wise forecast results
- explore charts such as line plots, bar charts, box plots, and pie charts
- compare current and forecasted sales

## Files in This Project

- `neww.py` - main Streamlit application with login/signup and sales forecasting
- `salesfinal.py` - advanced analytics dashboard with additional business intelligence features
- `bigmart.csv` - sample dataset used for testing
- `users.csv` - user credentials storage
- `requirements.txt` - Python dependencies

## Technologies Used

- Python
- Streamlit
- pandas
- numpy
- scikit-learn
- plotly
- XGBoost (optional)
- NeuralProphet (optional)
- bcrypt

## Installation

1. Clone or open the project folder.
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Run the App

Start the Streamlit app with:

```bash
streamlit run neww.py
```

You can also run the advanced dashboard using:

```bash
streamlit run salesfinal.py
```

## How It Works

1. The app asks the user to log in or create an account.
2. The user uploads a dataset in CSV or Excel format.
3. The app checks that the dataset contains a sales column such as `Item_Outlet_Sales`.
4. It preprocesses the data, encodes categorical features, and trains a regression model.
5. The app displays forecasting results and interactive charts.

## Dataset Requirements

The main forecasting workflow expects a dataset with at least:
- a sales target column such as `Item_Outlet_Sales`
- product/category columns such as `Item_Type`
- numeric features such as `Item_Weight`, `Item_MRP`, or `Item_Visibility`

## Notes

- If `xgboost` is installed, the app uses it for better prediction performance.
- If `neuralprophet` is installed, the app can generate an additional forecast visualization.
- The app stores simple user accounts in `users.csv`.

## License

This project is for educational and academic purposes.

