# Stock Analysis Dashboard

This repository contains the code for a Stock Analysis Dashboard built using Python, Dash, Plotly, and SQLite. The dashboard provides interactive tools for visualizing stock data, including time series plots, volume plots, trend analysis, volatility analysis, and various gauges representing stock performance over different time periods.
![image](https://github.com/user-attachments/assets/31502035-5b82-4f75-881d-2d1d096d5831 "Final Dashboard")

## Features

- **Category and Ticker Selection**: Filter stocks by category and select individual tickers to view detailed information and visualizations.
- **Stock Infobar**: Displays key information about the selected stock, including short name, current price, latest date, price change, and percentage change from the day prior to the latest date.
- **Gauges**: Visualize the stock's range levels and return levels over several timeframes (1 week, 1 month, 1 year, 5 years).
- **Time Series Plot**: View the stock's closing price over time with optional EMA (Exponential Moving Average) overlays.
- **Volume Plot**: Analyze the stock's trading volume with a 50-day moving average.
- **Trend Analysis**: Explore various trend metrics, including Open-Close percentage change, Bollinger Bands, MACD, RSI, and support/resistance levels.
- **Volatility Analysis**: Examine the stock's volatility through metrics such as high-low percentage change, ATR, Sharpe Ratio, and Beta.
- **Responsive UI**: The dashboard includes navigational buttons and dropdowns to switch between different stocks and customize visualizations.

## Structure

- **Dashboard v3.py**: The main file containing the Dash application logic, including callback functions for interactivity.
- **assets/**: Contains custom CSS files for styling the dashboard components.
- **NewStockData.db**: SQLite database file that stores stock data, including stock details, price data, and category mappings(not shared here due to file size, data source is yfinance and the code to build the database would be shared if requested).

## Key Components

### Callbacks

1. **update_tickers_and_selection**: 
   - Updates the ticker dropdown, infobar, gauges, and plots based on user interactions with the category dropdown, navigation buttons, or ticker selection.
   
2. **update_graph_based_on_buttons**: 
   - Updates time series and volume plots based on selected time ranges and EMA terms. Also highlights the active button.

3. **update_trend_plot**: 
   - Updates the trend and volatility plots based on selected periods and tab indices for more detailed analysis.

### Functions

- **get_categories**: Retrieves the list of available stock categories from the database.
- **get_tickers**: Fetches the list of tickers based on the selected category.
- **get_stock_data**: Retrieves detailed stock price data for the selected ticker.
- **get_ticker_infobar**: Populates the infobar with the latest stock information.
- **get_gauge_info**: Calculates and updates the gauge data based on historical stock performance.
- **create_time_series_plot**: Generates a Plotly figure for the stock's time series data, including optional EMA overlays.
- **create_volume_plot**: Produces a Plotly figure for the stock's trading volume.
- **create_trend_plot**: Visualizes different trend analysis metrics depending on the selected tab.
- **create_volatility_plot**: Displays various volatility metrics, including ATR, Sharpe Ratio, and Beta.

### `placeholder_config`

A configuration dictionary that stores settings and components for various UI placeholders in the dashboard, such as the infobar, search bar, category dropdown, gauges, and graphs.

## Setup

### Prerequisites

- Python 3.7 or higher
- Required Python packages:
  - `dash`
  - `dash-bootstrap-components`
  - `pandas`
  - `plotly`
  - `sqlite3`
  - `numpy`
  - `dateutil`

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/stock-analysis-dashboard.git
   cd stock-analysis-dashboard

2. Install the required packages:
   ```bash
   pip install -r requirements.txt

3. Place the NewStockData.db SQLite database in the root directory.

4. Run the application:
   ```bash
   python Dashboard v3.py
5. Open your web browser and navigate to http://127.0.0.1:8050/ to view the dashboard.


## Usage
- Use the Category Dropdown to filter stocks by category if needed.
- Navigate through stocks using the left and right buttons or the ticker dropdown.
- Select different time ranges and EMA terms to customize the plots.
- Explore detailed trend and volatility analysis through the interactive tabs, there is a dropdwon to select the time ranges separately for this section.

## Customization
You can customize the appearance and behavior of the dashboard by modifying the placeholder_config dictionary and the callback functions or the css file in assets.

## Contributing
If you'd like to contribute to this project, please fork the repository and create a pull request with your changes. I welcome all contributions that improve the functionality, usability, or appearance of the dashboard.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For questions or suggestions, feel free to contact the repository owner at [araju@mtu.edu].


