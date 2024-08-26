# Import necessary libraries
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc  # Bootstrap components for Dash
from dash.dependencies import Input, Output, State  # Add State here
import pandas as pd  # Pandas for data manipulation
import plotly.graph_objects as go  # Plotly for creating interactive plots
import sqlite3  # SQLite for database interactions
from datetime import datetime, timedelta  # DateTime for handling date and time
from dateutil.relativedelta import relativedelta  # For relative date calculations
import os  # OS module to handle file and directory paths
import numpy as np  # Numpy for numerical operations
##########--------------------##########--------------------##########--------------------
# Set the current working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
##########--------------------##########--------------------##########--------------------
# Function
def get_categories():
    # Function to retrieve stock categories from the database
    """
    NAME: get_categories
    DESCRIPTION: Fetches the list of all stock categories from the Category table in the SQLite database.
    PARAMETERS: None
    RETURNS: 
        - list: A list of category names as strings.
    """
    with sqlite3.connect('NewStockData.db') as conn:
        query = "SELECT category_name FROM Category"
        df = pd.read_sql(query, conn)  # Fetch data into a pandas DataFrame
    return df['category_name'].tolist() # Convert the DataFrame column to a list
def get_tickers(category=None):
    # Function to retrieve tickers based on a selected category or all tickers if no category is provided
    """
    NAME: get_tickers
    DESCRIPTION: Fetches the list of stock tickers from the Stock_Details table, optionally filtered by category.
    PARAMETERS:
        - category (str, optional): The category name to filter tickers by. Defaults to None.
    RETURNS: 
        - list: A list of ticker symbols as strings.
    """
    with sqlite3.connect('NewStockData.db') as conn:
        if category:
            query = """
            SELECT sd.ticker 
            FROM Stock_Details sd
            JOIN Stock_Categories sc ON sd.stock_id = sc.stock_id
            JOIN Category c ON sc.category_id = c.category_id
            WHERE c.category_name = ?
            ORDER BY sd.stock_id
            """
            df = pd.read_sql(query, conn, params=(category,)) # Fetch data with the category filter
        else:
            query = """
            SELECT ticker 
            FROM Stock_Details
            ORDER BY stock_id
            """
            df = pd.read_sql(query, conn) # Fetch all tickers without filtering by category
    
    return df['ticker'].tolist() # Convert the DataFrame column to a list
def get_date(date_str):
    # Function to format a date string from 'YYYY-MM-DD' to 'DD-MMM, YYYY'
    """
    NAME: get_date
    DESCRIPTION: Converts a date string from 'YYYY-MM-DD' format to 'DD-MMM, YYYY' format.
    PARAMETERS:
        - date_str (str): The date string in 'YYYY-MM-DD' format.
    RETURNS: 
        - str: The formatted date string in 'DD-MMM, YYYY' format.
    """
    # Convert the input date string to a datetime object
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    
    # Format the datetime object to 'DD-Jan, YYYY'
    formatted_date = date_obj.strftime('%d-%b, %Y')
    
    return f"on {formatted_date}" # Return the formatted date with "on" prefix
def get_returns(before, now):
    # Function to calculate the percentage returns between two values
    """
    NAME: get_returns
    DESCRIPTION: Calculates the percentage return between a previous value and a current value.
    PARAMETERS:
        - before (float or int): The previous value (can be None or 0).
        - now (float or int): The current value (can be None or 0).
    RETURNS: 
        - float: The percentage return value. Returns 0.1 if calculated return is 0.
    """
    if before is None or before == 0:
        before = 1e-6  # A small value close to zero to avoid division by zero
    if now is None or now == 0:
        now = 1e-6  # A small value close to zero to avoid division by zero
    f_returns = round((now - before)*100 / before) # Calculate the percentage return


    return 0.1 if f_returns == 0 else f_returns # Return 0.1 if return is 0, otherwise return the calculated value
def get_stock_data(ticker):
    # Function to retrieve stock data for a specific ticker from the database
    """
    NAME: get_stock_data
    DESCRIPTION: Fetches detailed stock data for a specific ticker from the Stock_Price table.
    PARAMETERS:
        - ticker (str): The stock ticker symbol for which data is to be fetched.
    RETURNS: 
        - pd.DataFrame: A pandas DataFrame containing stock data with columns like date, stock_open, stock_close, etc.
    """
    with sqlite3.connect('NewStockData.db') as conn:
        query = '''
        SELECT date, stock_open, stock_close, stock_high, stock_low, volume, 
               open_close_percent_change, high_low_percent_change, 
               volume_percent_change_min, volume_50_day_avg, daily_return, 
               daily_volatility, close_50_day_avg, ema_10_day, ema_20_day, 
               ema_50_day, ema_200_day, upper_band, lower_band, macd, 
               signal_line, rsi, atr, sharpe_ratio, beta
        FROM Stock_Price 
        WHERE stock_id = (SELECT stock_id FROM Stock_Details WHERE ticker = ?)
        '''
        data = pd.read_sql_query(query, conn, params=[ticker]) # Fetch data based on ticker
    return data   # Return the DataFrame with stock data
##########--------------------##########--------------------##########--------------------
# Global dictionary to store titles, labels, and functions for placeholders etc.
"""
    NAME: dash_data (Dictionary)
    DESCRIPTION: A global dictionary that stores various settings, configurations, and data used across the Dash application. 
                 This includes UI elements, colors, data for plots, selected options, etc.
    PARAMETERS: 
        - 'header': (str) The title for the dashboard.
        - 'category_list': (list) List of stock categories retrieved from the database.
        - 'selected_category': (str or None) The currently selected category.
        - 'ticker_list': (list) List of stock tickers retrieved from the database.
        - 'selected_ticker': (str or None) The currently selected stock ticker.
        - 'short_name': (str or None) Abbreviation or short name for the selected ticker.
        - 'current_price': (float or None) The latest stock price for the selected ticker.
        - 'date': (datetime or None) The date on the latest data for the ticker.
        - 'delta_price': (float or None) The price change between latest and date before that.
        - 'delta_percentage': (float or None) The percentage change between latest and date before that
        - 'ticker_colors': (dict or None) Color coding for positive and negative values based of delta_percentage.
        - 'range_level': (list of dicts) Stores min, max, start, end, and median values for range levels across different period ranges.
        - 'return_level': (list of dicts) Stores min, max, start, end, and median values for return levels across different period ranges.
        - 'selected_columns_line': (list or None) Columns selected for the line plot.
        - 'end_date': (datetime or None) The end date used for data selection for line and volume plot on left.
        - 'start_date': (datetime or None) The start date used for data selection for line and volume plot on left.
        - 'g2_start_date': (datetime or None) The start date used for data selection for line and volume plot on right.
        - 'time_series_data': (pd.DataFrame or None) Time series stock data for the selected ticker.
        - 'SPY_data': (pd.DataFrame or None) Time series stock data for the SPY ticker.
        - 'selected_button': (str) The button selection for the time period controlling the left plots
        - 'plot_bg_color': (str) The background color for the plot area.
        - 'paper_bg_color': (str) The background color for the entire figure.
        - 'plot_grid_color': (str) The color for grid lines in plots.
        - 'g2_title_color': (str) The color for titles in plots on the right.
        - 'annot_bg_color': (str) The background color for annotations.
        - 'stock_color': (str) The color for the stock price line.
        - 'stock_color_tint': (str) A tinted version of the stock color.
        - 'volume': (str) The color for the volume bars.
        - 'volume_tint': (str) A tinted version of the volume color.
        - 'color1', 'color1_tint', 'color2', 'color2_tint', 'color3', 'color3_tint': (str) Additional colors and tints for different EMA lines and so.
        - 'line_width': (list) List of line widths for different plots.
        - 'text_color': (str) The color for text in plots.
        - 'selection_color': (str) The color for selected elements.
        - 'selection_border_color': (str) The border color for selected elements.
        - 'selected_tab_top': (int or None) Index of the selected tab in the top right section of the dashboard.
        - 'selected_tab_bottom': (int or None) Index of the selected tab in the bottom right section of the dashboard.
    RETURNS: None
"""
dash_data = {
    'header': 'Stock Analysis Dashboard',
    'category_list': get_categories(),
    'selected_category': None,
    'ticker_list': get_tickers(),
    'selected_ticker': None,
    'short_name': None,
    'current_price': None,
    'date': None,
    'delta_price': None,
    'delta_percentage': None,
    'ticker_colors': None,
    #'before_value':[None] * 4,
    #'spy_value':[None] * 4,
    'range_level': [{'min': None, 'max': None, 'start': None, 'end': None, 'median': None} for _ in range(4)],
    'return_level': [{'min': None, 'max': None, 'start': None, 'end': None, 'median': None} for _ in range(4)],
    'selected_columns_line':['ema_10_day', 'ema_50_day'],
    'end_date': None,
    'start_date':None,
    'g2_start_date': None,
    'time_series_data': None,
    'SPY_data': get_stock_data('SPY'),
    'selected_button': 'btn-3M',
    'plot_bg_color' : '#FAF9F6', #FAF9F6 for off white
    'paper_bg_color' : '#ffffff', #pure white
    'plot_grid_color' : '#ededed', # lighter grey
    'g2_title_color' : '#9a9a9a', # shade of black
    'annot_bg_color' : 'rgba(250,249,246,0.7)', #plot bg color with some transparency to see plots behind
    'stock_color' : 'rgba(51, 51, 153,1)', #shade of blue
    'stock_color_tint' : 'rgba(51, 51, 153,0.5)', #stock_color with opaque
    'volume' : '#8A2BE2', #Blue Violet for volume
    'volume_tint' : '#e9d8fa', #Volume color with 0.1 opacity
    'color1' : '#993366',
    'color1_tint' : '#D47EA9',
    'color2' : '#999933',
    'color2_tint' : '#D4D47E',
    'color3' : '#339966',
    'color3_tint' : '#7ED4A9',
    'line_width' : [3,1.5, 2,1], 
    'text_color': 'grey',
    'selection_color': '#ebebff',
    'selection_border_color': '#8989ff',
    'selected_tab_top': None,
    'selected_tab_bottom': None,
}
dash_data['selected_ticker'] = dash_data['ticker_list'][0]
dash_data['time_series_data'] = get_stock_data(dash_data['selected_ticker'])
dash_data['time_series_data']['date'] =  pd.to_datetime(dash_data['time_series_data']['date'])
dash_data['SPY_data']['date'] =  pd.to_datetime(dash_data['SPY_data']['date'])
dash_data['end_date'] = dash_data['time_series_data']['date'].max()
dash_data['start_date'] = dash_data['end_date'] - pd.DateOffset(months=3) #selected 3 months randomly from the available button options
dash_data['g2_start_date'] = dash_data['start_date']
dash_data['selected_tab_top'] = 0
dash_data['selected_tab_bottom'] = 0
tick_colors = {
    'Positive': 'black',
    'Negative': 'red'
}
##########--------------------##########--------------------##########--------------------
# Creating the div elements
def create_category_dropdown():
    # Function to create the category dropdown UI element
    """
    NAME: create_category_dropdown
    DESCRIPTION: Creates a Dash Dropdown component for selecting stock categories.
    PARAMETERS: None
    RETURNS: 
        - dcc.Dropdown: A Dash Dropdown component configured with available categories.
    """
    return dcc.Dropdown(
        id='category-dropdown',
        options=[{'label': category, 'value': category} for category in dash_data['category_list']],
        value = dash_data['selected_category'] if dash_data['selected_category'] else None,
        placeholder='Filter stocks by category',
        style={'width': '100%', 'height': '5vh'}  # Define width and height here
    )
def create_ticker_dropdown():
    # Function to create the ticker dropdown UI element
    """
    NAME: create_ticker_dropdown
    DESCRIPTION: Creates a Dash Dropdown component for selecting stock tickers.
    PARAMETERS: None
    RETURNS: 
        - dcc.Dropdown: A Dash Dropdown component configured with available tickers.
    """
    return dcc.Dropdown(
        id='ticker-dropdown',
        options=[{'label': ticker, 'value': ticker} for ticker in dash_data['ticker_list']],
        value=dash_data['selected_ticker'] if dash_data['selected_ticker'] else None,  # Set the default value
        style={
            'width': '100%', 
            'height': '5vh', 
            'textAlign': 'center',  # Center the text
            'color': 'black'  # Set text color to black
        },
        clearable=False  # Ensure that the dropdown does not allow clearing to show the default value
    )
def create_left_button():
    # Function to create the left navigation button
    """
    NAME: create_left_button
    DESCRIPTION: Creates an HTML Button element for left navigation (to cycle through tickers).
    PARAMETERS: None
    RETURNS: 
        - html.Button: A Dash Button element styled for left navigation.
    """
    return html.Button('<', id='left-button', style={'height': '100%', 'width': '100%', 'border': '1px solid {}'.format(dash_data['selection_border_color']), 'background-color': dash_data['selection_color'], 'color':dash_data['selection_border_color'], 'margin':'0', 'padding':'0', 'font-size': '3vh'})
def create_right_button():
    # Function to create the right navigation button
    """
    NAME: create_right_button
    DESCRIPTION: Creates an HTML Button element for right navigation (to cycle through tickers).
    PARAMETERS: None
    RETURNS: 
        - html.Button: A Dash Button element styled for right navigation.
    """
    return html.Button('>', id='right-button', style={'height': '100%', 'width': '100%', 'border': '1px solid {}'.format(dash_data['selection_border_color']), 'background-color': dash_data['selection_color'], 'color':dash_data['selection_border_color'], 'margin':'0', 'padding':'0', 'font-size': '3vh'})
def create_time_series_plot(data, columns, plot_type='line'):
    # Function to create a time series plot with no volume plot on a secondary axis
    """
    NAME: create_time_series_plot
    DESCRIPTION: Generates a Plotly figure displaying stock closing prices and optional EMAs.
    PARAMETERS:
        - data (pd.DataFrame): The stock data, including columns like date, stock_close, volume, etc.
        - columns (list): A list of EMA columns to plot, e.g., ['ema_10_day', 'ema_50_day'].
        - plot_type (str, optional): The type of plot for the stock_close data ('line', 'mountain', or 'candlestick'). Default is 'line'.
    RETURNS: 
        - go.Figure: A Plotly figure object configured for time series analysis.
    """
    fig = go.Figure()

    # Plot type for stock_close
    if plot_type == 'line':
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['stock_close'],
            mode='lines',
            name='Stock Close',
            hovertemplate=f'<span style="font-weight:bold; font-size:1vw;">{dash_data["selected_ticker"]}' +  '<span style="font-weight:bold; font-size:1vw;"> Close : $%{y:.1f}</span><extra></extra>',
            line=dict(color=dash_data['stock_color'], width = dash_data['line_width'][0]),  # Blue
            #opacity=1  # 100% opacity
        ))
    elif plot_type == 'mountain':
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['stock_close'],
            mode='lines',
            fill='tozeroy',
            name='Stock Close',
            hovertemplate=f'<span style="font-weight:bold; font-size:1vw;">{dash_data["selected_ticker"]}' +  '<span style="font-weight:bold; font-size:1vw;"> Close : $%{y:.1f}</span><extra></extra>',
            line=dict(color=dash_data['stock_color'], width = dash_data['line_width'][0]),  # Blue
            #opacity=1  # 100% opacity
        ))
    elif plot_type == 'candlestick':
        fig.add_trace(go.Candlestick(
            x=data['date'],
            open=data['stock_open'],
            high=data['stock_high'],
            low=data['stock_low'],
            close=data['stock_close'],
            name='',
            #hovertext='Open: $%{y:.1f}<br>High: $%{y:.1f}<br>Low: $%{y:.1f}<br>Close: $%{y:.1f}<extra></extra>'
        ))

    # Loop through the list of EMA columns
    for ema_col in columns:
        if ema_col == 'ema_10_day':
            line_color = dash_data['color1']  # Orange
            line_type = 'dash'
            line_width = dash_data['line_width'][1]
        elif ema_col == 'ema_20_day':
            line_color = dash_data['color1_tint']  # Orange
            line_type = 'solid'
            line_width = dash_data['line_width'][1]
        elif ema_col == 'ema_50_day':
            line_color = dash_data['color2']  # Green
            line_type = 'dash'
            line_width = dash_data['line_width'][2]
        elif ema_col == 'ema_200_day':
            line_color = dash_data['color2_tint']  # Lime Green
            line_type = 'solid'
            line_width = dash_data['line_width'][2]

        if ema_col in data.columns:
            fig.add_trace(go.Scatter(
                x=data['date'],
                y=data[ema_col],
                mode='lines',
                name=ema_col.replace('_', ' ').title(),
                hovertemplate=f'{ema_col.replace("_", " ").title()}: $%{{y:.1f}}<extra></extra>',
                line=dict(color=line_color, dash=line_type, width = line_width),  # Using the variables defined in the if loop
                #opacity=1  # 100% opacity
            ))

    # Volume bar plot on secondary y-axis
    fig.add_trace(go.Bar(
        x=data['date'],
        y=data['volume'],
        name='Volume',
        yaxis='y2',
        marker=dict(color=dash_data['volume']),  # Custom color
        opacity=0.1,  # 30% opacity
        hovertemplate='Volume: %{y:.1s}<extra></extra>'
    ))

    # Calculate the stock close price difference and percentage difference
    if not data.empty:
        start_price = data[data['date'] == dash_data['start_date']]['stock_close'].values[0]
        end_price = data[data['date'] == dash_data['end_date']]['stock_close'].values[0]
        price_diff = end_price - start_price
        percent_diff = (price_diff / start_price) * 100

        # Set the font color based on the price difference
        font_color = 'green' if price_diff >= 0 else 'red'

        # Add the annotation box at the top left corner
        fig.add_annotation(
            #xref="x", yref="y",  # Refers to the data coordinates
            #x=dash_data['end_date'],  # The x-coordinate (date) of the last stock_close price
            #y=end_price,  # The y-coordinate (value) of the last stock_close price
            xref="paper", yref="paper",
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
            text=(
                f"<b style='color:grey; font-size:1vw;'>$ Diff: </b> "
                f"<span style='color:{font_color}; font-weight:bold; font-size:1vw;'>${price_diff:.2f}</span><br><br>"
                f"<b style='color:grey; font-size:1vw;'>% Diff: </b> "
                f"<span style='color:{font_color}; font-weight:bold; font-size:1vw;'>{percent_diff:.2f}%</span>"
            ),
            showarrow=False,
            #arrowhead=8,  # Arrow style (1-8 are available styles)
            #arrowsize=1,  # Arrow size multiplier
            #arrowwidth=2,  # Arrow line width
            #ax=-250,  # Offset of the text relative to the arrow (horizontal)
            #ay=-200,  # Offset of the text relative to the arrow (vertical)
            align="left",
            bordercolor=font_color,
            borderwidth=0,
            borderpad=0,
            bgcolor= dash_data['annot_bg_color'],
            opacity=1
            )

    fig.update_layout(
        yaxis2=dict(overlaying='y', side='right', showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, gridcolor=dash_data['plot_grid_color']),
        xaxis=dict(showticklabels=True, tickfont=dict(color=dash_data['text_color'])),
        xaxis_rangeslider_visible=False,  # Removes the range slider that might show volume
        plot_bgcolor=dash_data['plot_bg_color'], 
        paper_bgcolor= dash_data['paper_bg_color'],
        hovermode='x unified',
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        modebar_remove=['zoom', 'zoomin', 'zoomout', 'pan', 'select', 'lasso', 'resetScale2d', 'toimage'],
        modebar_add=['autoscale'],
        modebar=dict(
            orientation='h',
            bgcolor='rgba(0, 0, 0, 0)',
            color='lightgrey',
            activecolor='grey'
        )
    )

    return fig
def create_volume_plot(data):
    # Function to create a volume plot with 50-day average
    """
    NAME: create_volume_plot
    DESCRIPTION: Generates a Plotly figure displaying the stock volume and its 50-day moving average.
    PARAMETERS:
        - data (pd.DataFrame): The stock data, including columns like date, volume, and volume_50_day_avg.
    RETURNS: 
        - go.Figure: A Plotly figure object configured for volume analysis.
    """
    fig = go.Figure()

    # Volume bar plot
    fig.add_trace(go.Bar(
        x=data['date'],
        y=data['volume'],
        name='Volume',
        hovertemplate='Volume: %{y:.1s}<extra></extra>',
        marker=dict(color=dash_data['volume']),  # Custom color
        opacity=0.3  # 30% opacity
    ))

    # Line plot for volume_50_day_avg
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['volume_50_day_avg'],
        mode='lines',
        name='Volume 50-Day Avg',
        hovertemplate='Volume 50-Day Avg: %{y:.1s}<extra></extra>',
        line=dict(color=dash_data['volume']),  # Pink
        opacity=0.8  # 80% opacity
    ))

    fig.update_layout(
        # Hide y-axis tick labels
        #title=f"{ticker} Volume Data",
        #yaxis=dict(title='Volume'),
        #xaxis=dict(title='Date'),
        yaxis=dict(showticklabels=False, gridcolor=dash_data['plot_grid_color']),
        xaxis=dict(showticklabels=False),
        # Remove plot margins
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor=dash_data['plot_bg_color'],  # Background color of the plot area
        paper_bgcolor=dash_data['paper_bg_color'],  # Background color of the entire figure
        # Change the location of the legend to the top right inside corner
        legend=dict(
            x=0,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor=dash_data['annot_bg_color'],  # Transparent background
            bordercolor='rgba(0, 0, 0, 0)',  # Transparent border
            borderwidth=0,
        ),
        hovermode='x unified',
        # Hide plotly options except autoscale
        modebar_remove=['zoom', 'zoomin', 'zoomout', 'pan', 'select', 'lasso', 'resetScale2d', 'toimage'],
        modebar_add=['autoscale'],
        modebar=dict(
            orientation='h',  # Vertical orientation
            bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            color='lightgrey',  # Button color
            activecolor='grey'  # Active button color
        )
    )
    
    # Uncomment the next line to use log scale for y-axis
    # fig.update_yaxes(type="log")

    return fig
def create_trend_plot(data, tab):
    # Function to create various trend analysis plots based on the selected tab
    """
    NAME: create_trend_plot
    DESCRIPTION: Generates a Plotly figure for different types of trend analysis (Open-Close % Change, Histogram, Bollinger Bands, MACD, RSI, or Support and Resistance) based on the selected tab.
    PARAMETERS:
        - data (pd.DataFrame): The stock data, including columns relevant to the selected analysis.
        - tab (int): The selected tab index (0-5), each corresponding to a specific type of analysis.
    RETURNS: 
        - go.Figure: A Plotly figure object configured for the selected trend analysis.
    """
    fig = go.Figure()
    #line=dict(color=line_color, dash=line_type, width = line_width),  # Using the variables defined in the if loop
    
    if tab == 0:
        # Open-Close % Change Line plot
        fig.add_trace(go.Scatter(
            x=data['date'], 
            y=data['open_close_percent_change'], 
            mode='lines',
            name=f"{dash_data['selected_ticker']}" + ' Open-Close % Change',
            line=dict(color=dash_data['stock_color'], width = dash_data['line_width'][2]),  # Blue
            hovertemplate=f"{dash_data['selected_ticker']}" + ' % Change: %{y:.2f}%<extra></extra>'
        ))
        average_level = np.mean(data['open_close_percent_change'].values)
        fig.add_hline(y=average_level, line=dict(color="red", width=2, dash="dash"), name='Average Level')

        # Add annotation to show the average level on the plot
        fig.add_annotation(
            x=data['date'].iloc[len(data)//2],  # Show annotation at the last date
            y=average_level,
            #text=f"Average: {average_level:.2f}%",
            text=(
                f"<span style='color:red;  font-size:0.8vw;'>Average: {average_level:.2f}%</span><br><br>"
            ),
            showarrow=False,
            font=dict(color="red"),
            align="right",
            xanchor="right",
            yanchor="bottom",
            xshift=10,
            bgcolor=dash_data['annot_bg_color'],
            opacity=1
        )
        fig.update_layout(
            showlegend=False,
        )
        custom_title = "Daily Percentage Change: Opening to Closing Price"
        dash_data['selected_tab_top'] = 0

    elif tab == 1:
        # Calculate histogram data and CDF
        hist_data = np.histogram(data['open_close_percent_change'], bins=100)
        cdf = np.cumsum(hist_data[0]) / np.sum(hist_data[0])
        
        # Open-Close % Change Histogram Plot        
        fig.add_trace(go.Histogram(
            x=data['open_close_percent_change'],
            name='',
            marker=dict(color=dash_data['stock_color_tint'], line=dict(color=dash_data['stock_color'], width=1)),
            nbinsx=30,  # Number of bins, can adjust as needed
            histnorm='probability density',  # Normalize histogram to show density
            hovertemplate=(
                f"{dash_data['selected_ticker']}" + ' % Change Range : %{x}% <extra></extra>'
            ),
        ))

        # Add CDF line plot on secondary y-axis
        fig.add_trace(go.Scatter(
            x=hist_data[1][:],  # Bin edges (excluding the first edge, which is the start of the first bin)
            y=cdf,
            name='CDF',
            yaxis='y2',
            mode='lines',
            line=dict(color='red', width=2),
            customdata = cdf*100,
            hovertemplate=(
                f"{dash_data['selected_ticker']}"+' Cumulative Probability : %{customdata:.1f}% <extra></extra>'
            ),
        ))

        # Update layout to include a secondary y-axis
        fig.update_layout(
            #title="Open-Close % Change Histogram with CDF",
            #xaxis_title="Open-Close % Change",
            #yaxis_title="Probability Density",
            yaxis2=dict(
                #title="CDF",
                overlaying='y',
                side='right',
                showgrid=False,
                showticklabels=False
            ),
            #legend=dict(x=0.8, y=0.9),
            showlegend=False,
        )
        custom_title = "Distribution of Daily Percentage Change: Open to Close Price"
        dash_data['selected_tab_top'] = 1
        
    elif tab == 2:
        # Bollinger Bands for Closing Price
        fig.add_trace(go.Scatter(
            x=data['date'], 
            y=data['close_50_day_avg'], 
            mode='lines',
            name=f"{dash_data['selected_ticker']}" + ' 50-Day MA',
            line=dict(color=dash_data['stock_color'], width = dash_data['line_width'][2]),  # Blue
            hovertemplate=(
                f"{dash_data['selected_ticker']}" + ' 50-Day MA : $%{y:.1f} <extra></extra>'
            ),
        ))
        fig.add_trace(go.Scatter(
            x=data['date'], 
            y=data['upper_band'], 
            mode='lines',
            fill=None,
            name='Upper Band',
            line=dict(color='red', width = dash_data['line_width'][3]),  # Blue
            hovertemplate=(
                'Upper : $%{y:.1f} <extra></extra>'
            ),
        ))
        fig.add_trace(go.Scatter(
            x=data['date'], 
            y=data['lower_band'], 
            mode='lines',
            fill='tonexty',  # fill area between upper and lower bands
            name='Lower Band',
            line=dict(color='#339933', width = dash_data['line_width'][3]),  # Blue
            hovertemplate=(
                'Lower : $%{y:.1f} <extra></extra>'
            ),
        ))
        custom_title = "Bollinger Bands Analysis: Daily Closing Price"
        fig.update_layout(
            legend=dict(
                x=0.01,  # Position the legend inside the plot area
                y=0.99,
                xanchor='left',
                yanchor='top',
                bgcolor=dash_data['annot_bg_color'],  # Same as plot_bgcolor with controlled opacity
                bordercolor='rgba(0, 0, 0, 0)',  # Transparent border
            ),
        )
        dash_data['selected_tab_top'] = 2
    
    elif tab == 3:
        # MACD Plot
        fig.add_trace(go.Scatter(
            x=data['date'], 
            y=data['macd'], 
            mode='lines',
            name=f"{dash_data['selected_ticker']}" + ' MACD',
            line=dict(color=dash_data['stock_color'], width = dash_data['line_width'][2]),  # Blue
        ))
        fig.add_trace(go.Scatter( 
            x=data['date'], 
            y=data['signal_line'], 
            mode='lines',
            name=f"{dash_data['selected_ticker']}"+' Signal Line',
            line=dict(color='red', width = dash_data['line_width'][1], dash = 'dash'),  # Blue
        ))
        # MACD Histogram (Bar Plot)
        '''
        fig.add_trace(go.Bar(
            x=data['date'], 
            y=data['macd']-data['signal_line'], 
            name='MACD Histogram',
            marker=dict(color=(data['macd']-data['signal_line']).apply(lambda x: 'green' if x >= 0 else 'red')),
            opacity=0.5
        ))
        '''
        # Volume (Bar Plot)
        fig.add_trace(go.Bar(
            x=data['date'], 
            y=data['volume'], 
            name=f"{dash_data['selected_ticker']}" + ' Volume',
            marker=dict(color=dash_data['volume']),  # Custom color
            yaxis='y2',
            opacity=0.1,  # 30% opacity
            hovertemplate=f"{dash_data['selected_ticker']}" + ' Volume : %{y:.1s}<extra></extra>'
        ))

        custom_title = "MACD Analysis: Momentum and Trend Indicator"
        fig.update_layout(
            legend=dict(
                x=0.01,  # Position the legend inside the plot area
                y=0.99,
                xanchor='left',
                yanchor='top',
                bgcolor=dash_data['annot_bg_color'],  # Same as plot_bgcolor with controlled opacity
                bordercolor='rgba(0, 0, 0, 0)',  # Transparent border
            ),
            yaxis2=dict(
                #title='Volume',
                overlaying='y',
                side='right',
                showgrid=False,
                showticklabels=False
            ),
        )
        dash_data['selected_tab_top'] = 3
    
    elif tab == 4:
        # RSI Plot
        fig.add_trace(go.Scatter(
            x=data['date'], 
            y=data['rsi'], 
            mode='lines',
            name=f"{dash_data['selected_ticker']}" + ' RSI',
            line=dict(color=dash_data['stock_color'], width = dash_data['line_width'][2]),  # Blue
            hovertemplate=f"{dash_data['selected_ticker']}" + ' RSI : %{y:.0f}%<extra></extra>'
        ))

        # Adding a trace for stock close price
        fig.add_trace(go.Scatter(
            x=data['date'], 
            y=data['stock_close'], 
            mode='lines',
            name=f"{dash_data['selected_ticker']}" + ' Close Price',
            yaxis='y2',  # Assigning a secondary y-axis if you want to use it
            line=dict(color=dash_data['color2'], width = dash_data['line_width'][3]),  # Blue
            #opacity=0.3,
            hovertemplate=f"{dash_data['selected_ticker']}" + ' Close Price : $%{y:.2f}<extra></extra>'
        ))

        fig.add_hline(y=70, line=dict(color="red", width=2, dash="dash"), name='Overbought Level')
        fig.add_hline(y=30, line=dict(color="green", width=2, dash="dash"), name='Oversold Level')

        # Customizing the layout for secondary y-axis (if using it)
        fig.update_layout(
            #title=custom_title,
            yaxis2=dict(
                #title="Stock Close Price",
                overlaying='y',  # overlaying the secondary y-axis on the primary y-axis
                side='right',
                showgrid=False,
                showticklabels=False
            ),
            legend=dict(
                x=0.01,  # Position the legend inside the plot area
                y=0.99,
                xanchor='left',
                yanchor='top',
                bgcolor=dash_data['annot_bg_color'],  # Same as plot_bgcolor with controlled opacity
                bordercolor='rgba(0, 0, 0, 0)',  # Transparent border
            )
        )

        fig.add_annotation(
            x=data['date'].iloc[len(data)//2],  # Show annotation at the last date
            y=70,
            #text=f"Average: {average_level:.2f}%",
            text=(
                f"<span style='color:red;  font-size:0.8vw;'>RSI > 70% : Overbought</span><br><br>"
            ),
            showarrow=False,
            font=dict(color="red"),
            align="right",
            xanchor="right",
            yanchor="bottom",
            xshift=10,
            bgcolor=dash_data['annot_bg_color'],
            opacity=1
        )
        
        fig.add_annotation(
            x=data['date'].iloc[len(data)//2],  # Show annotation at the last date
            y=30,
            #text=f"Average: {average_level:.2f}%",
            text=(
                f"<span style='color:green;  font-size:0.8vw;'>RSI < 30% : Oversold </span><br><br>"
            ),
            showarrow=False,
            font=dict(color="green"),
            align="right",
            xanchor="right",
            yanchor="bottom",
            xshift=10,
            bgcolor=dash_data['annot_bg_color'],
            opacity=1
        )
        
        custom_title = "RSI Analysis: Relative Strength Index"
        dash_data['selected_tab_top'] = 4
    
    elif tab == 5:
        # Support and Resistance Levels
        fig.add_trace(go.Scatter(
            x=data['date'], 
            y=data['close_50_day_avg'], 
            mode='lines',
            name=f"{dash_data['selected_ticker']}" + ' 50-Day MA',
            line=dict(color=dash_data['stock_color'], width = dash_data['line_width'][2]),  # Blue
            hovertemplate=f"{dash_data['selected_ticker']}" + ' 50-Day MA : $%{y:.1f}<extra></extra>'
        ))
        # Example support and resistance levels (customize as needed)
        support_level = data['close_50_day_avg'].min()
        resistance_level = data['close_50_day_avg'].max()

        fig.add_hline(y=support_level, line=dict(color="green", width=2, dash="dash"), name='Support Level')
        fig.add_hline(y=resistance_level, line=dict(color="red", width=2, dash="dash"), name='Resistance Level')

        fig.add_annotation(
            x=data['date'].iloc[len(data)//2],  # Show annotation at the last date
            y=resistance_level,
            #text=f"Average: {average_level:.2f}%",
            text=(
                f"<span style='color:red;  font-size:0.8vw;'>Resistance Level</span><br><br>"
            ),
            showarrow=False,
            font=dict(color="red"),
            align="right",
            xanchor="right",
            yanchor="bottom",
            xshift=10,
            bgcolor=dash_data['annot_bg_color'],
            opacity=1
        )
        
        fig.add_annotation(
            x=data['date'].iloc[len(data)//2],  # Show annotation at the last date
            y=support_level,
            #text=f"Average: {average_level:.2f}%",
            text=(
                f"<span style='color:green;  font-size:0.8vw;'>Support Level</span><br><br>"
            ),
            showarrow=False,
            font=dict(color="green"),
            align="right",
            xanchor="right",
            yanchor="bottom",
            xshift=10,
            bgcolor=dash_data['annot_bg_color'],
            opacity=1
        )
        

        custom_title = "Support and Resistance Analysis: Key Price Levels"
        fig.update_layout(
            showlegend=True,
            legend=dict(
                x=0.01,  # Position the legend inside the plot area
                y=0.99,
                xanchor='left',
                yanchor='top',
                bgcolor=dash_data['annot_bg_color'],  # Same as plot_bgcolor with controlled opacity
                bordercolor='rgba(0, 0, 0, 0)',  # Transparent border
            ),
        )
        dash_data['selected_tab_top'] = 5
    
    else:
        raise ValueError("Invalid tab value. Please select a valid tab (0-5).")

    # Add custom title annotation inside the plot area
    fig.add_annotation(
        #text=custom_title,                   # Custom title text
        text=(
            f"<span style='color:{dash_data['g2_title_color']}; font-weight:normal;  font-variant:small-caps; font-size:0.8vw;'>{custom_title}</span><br><br>"
        ),
        xref="paper", yref="paper",          # Reference the plot area
        x=0.5, y=1,                        # Position (x, y) inside the plot area (0.5, 0.9) corresponds to center-top
        xanchor="center", yanchor="top",
        yshift=20,
        showarrow=False,                     # No arrow pointing to the text
        align="center",                       # Center-align the text
        borderwidth=0,
        borderpad=0,
        bgcolor=dash_data['paper_bg_color'],
        opacity=1
    )

    fig.update_layout(
        yaxis=dict(showticklabels=False, gridcolor=dash_data['plot_grid_color']),
        xaxis=dict(showgrid=False, tickfont=dict(color=dash_data['text_color'])),
        plot_bgcolor=dash_data['plot_bg_color'],
        paper_bgcolor=dash_data['paper_bg_color'],
        hovermode='x unified',
        margin=dict(l=0, r=0, t=20, b=0),
        #showlegend=False,
        modebar_remove=['zoom', 'zoomin', 'zoomout', 'pan', 'select', 'lasso', 'resetScale2d', 'toimage'],
        modebar_add=['autoscale'],
        modebar=dict(
            orientation='h',
            bgcolor='rgba(0, 0, 0, 0)',
            color='lightgrey',
            activecolor='grey'
        )
    )
    return fig
def create_volatility_plot(data, tab, data2 = dash_data['SPY_data']):
    # Function to create various volatility analysis plots based on the selected tab
    """
    NAME: create_volatility_plot
    DESCRIPTION: Generates a Plotly figure for different types of volatility analysis (High-Low % Change, ATR, Sharpe Ratio, Volatility, or Beta) based on the selected tab.
    PARAMETERS:
        - data (pd.DataFrame): The stock data, including columns relevant to the selected analysis.
        - tab (int): The selected tab index (0-5), each corresponding to a specific type of analysis.
        - data2 (pd.DataFrame, optional): The SPY data for comparison in certain plots, such as Sharpe Ratio and Beta.
    RETURNS: 
        - go.Figure: A Plotly figure object configured for the selected volatility analysis.
    """
    fig = go.Figure()

    if tab == 0:
        # High-Low % Change Line plot
        fig.add_trace(go.Scatter(
            x=data['date'], 
            y=data['high_low_percent_change'], 
            mode='lines',
            name=f"{dash_data['selected_ticker']}" + ' High-Low % Change',
            line=dict(color=dash_data['stock_color'], width = dash_data['line_width'][2]),  # Blue
            hovertemplate=f"{dash_data['selected_ticker']}" + ' % Change: %{y:.2f}%<extra></extra>'
        ))
        average_level = np.mean(data['high_low_percent_change'].values)
        fig.add_hline(y=average_level, line=dict(color="red", width=2, dash="dash"), name='Average Level')
        # Add annotation to show the average level on the plot
        fig.add_annotation(
            x=data['date'].iloc[len(data)//2],  # Show annotation at the mid date
            y=average_level,
            #text=f"Average: {average_level:.2f}%",
            text=(
                f"<span style='color:red;  font-size:0.8vw;'>Average: {average_level:.2f}%</span><br><br>"
            ),
            showarrow=False,
            font=dict(color="red"),
            align="right",
            xanchor="right",
            yanchor="bottom",
            xshift=10,
            bgcolor=dash_data['annot_bg_color'],
            opacity=1
        )

        custom_title = "Daily Percentage Change: Low to High Price"

        fig.update_layout(
            showlegend=False,
        )
        dash_data['selected_tab_bottom'] = 0
 
    elif tab == 1:
        # Calculate histogram data and CDF
        hist_data = np.histogram(data['high_low_percent_change'], bins=100)
        cdf = np.cumsum(hist_data[0]) / np.sum(hist_data[0])
        
        # High-Low % Change Histogram Plot
        fig.add_trace(go.Histogram(
            x=data['high_low_percent_change'],
            name='',
            marker=dict(color=dash_data['stock_color_tint'], line=dict(color=dash_data['stock_color'], width=1)),
            nbinsx=30,  # Number of bins, can adjust as needed
            histnorm='probability density',  # Normalize histogram to show density
            hovertemplate=(
                f"{dash_data['selected_ticker']}" + ' % Change Range : %{x}% <extra></extra>'
            ),
        ))

        # Add CDF line plot on secondary y-axis
        fig.add_trace(go.Scatter(
            x=hist_data[1][:],  # Bin edges (excluding the first edge, which is the start of the first bin)
            y=cdf,
            name='CDF',
            yaxis='y2',
            mode='lines',
            line=dict(color='red', width=2),
            customdata = cdf*100,
            hovertemplate=(
                f"{dash_data['selected_ticker']}" + ' Cumulative Probability : %{customdata:.1f}% <extra></extra>'
            ),
        ))
        # Update layout to include a secondary y-axis
        fig.update_layout(
            #title="Open-Close % Change Histogram with CDF",
            #xaxis_title="Open-Close % Change",
            #yaxis_title="Probability Density",
            yaxis2=dict(
                #title="CDF",
                overlaying='y',
                side='right',
                showgrid=False,
                showticklabels=False
            ),
            #legend=dict(x=0.8, y=0.9),
            showlegend=False,
        )
        custom_title = "Distribution of Daily Percentage Change: Low to High Price"
        dash_data['selected_tab_bottom'] = 1

    elif tab == 2:
        # ATR (Average True Range) Plot
        fig.add_trace(go.Scatter(
            x=data['date'], 
            y=data['atr'], 
            mode='lines',
            name=f"{dash_data['selected_ticker']}"+' ATR',
            line=dict(color=dash_data['stock_color'], width = dash_data['line_width'][2]),  # Blue
            hovertemplate=f"{dash_data['selected_ticker']}" + ' ATR : $%{y:.2f}<extra></extra>'
        ))
        # Volume (Bar Plot)
        fig.add_trace(go.Bar(
            x=data['date'], 
            y=data['volume'], 
            name=f"{dash_data['selected_ticker']}"+' Volume',
            marker=dict(color=dash_data['volume']),  # Custom color
            yaxis='y2',
            opacity=0.1,
            hovertemplate=f"{dash_data['selected_ticker']}" + ' Volume : %{y:.1s}<extra></extra>'
        ))
        fig.update_layout(
            legend=dict(
                x=0.01,  # Position the legend inside the plot area
                y=0.99,
                xanchor='left',
                yanchor='top',
                bgcolor=dash_data['annot_bg_color'],  # Same as plot_bgcolor with controlled opacity
                bordercolor='rgba(0, 0, 0, 0)',  # Transparent border
            ),
            yaxis2=dict(
                #title='Volume',
                overlaying='y',
                side='right',
                showgrid=False,
                showticklabels=False
            ),
        )
        custom_title = "Trend Strength & Volatility: 14-Day ATR"
        dash_data['selected_tab_bottom'] = 2
    
    elif tab == 3:
        # Sharpe Ratio Over Time
        fig.add_trace(go.Scatter(
            x=data['date'], 
            y=data['sharpe_ratio'], 
            mode='lines',
            name=f"{dash_data['selected_ticker']} Sharpe Ratio",
            line=dict(color=dash_data['stock_color'], width = dash_data['line_width'][2]),  # Blue
            hovertemplate=f"{dash_data['selected_ticker']}"+' : %{y:.2f}<extra></extra>',
        ))
        # add line plot for S&P500  sharpe ratio for comparison
        fig.add_trace(go.Scatter(
            x=data['date'], 
            y=data2['sharpe_ratio'], 
            mode='lines',
            name='SPY Sharpe Ratio',
            line=dict(color=dash_data['color1'], width = dash_data['line_width'][3]),  # Blue
            hovertemplate='SPY : %{y:.2f}<extra></extra>'
        ))
        fig.update_layout(
            legend=dict(
                x=0.01,  # Position the legend inside the plot area
                y=0.99,
                xanchor='left',
                yanchor='top',
                bgcolor= dash_data['annot_bg_color'],  # Same as plot_bgcolor with controlled opacity
                bordercolor='rgba(0, 0, 0, 0)',  # Transparent border
            ),
        )
        custom_title = "Returns & Volatility Analysis: 50-Week Sharpe Ratio"
        dash_data['selected_tab_bottom'] = 3
    
    elif tab == 4:
        # Volatility Plot
        fig.add_trace(go.Scatter(
            x=data['date'], 
            y=data['daily_volatility'], 
            mode='lines',
            name=f"{dash_data['selected_ticker']} Volatility",
            line=dict(color=dash_data['stock_color'], width = dash_data['line_width'][2]),  # Blue
            hovertemplate=f"{dash_data['selected_ticker']}" + ' Volatility : %{y:.2f}%<extra></extra>'
        ))
        # Volume (Bar Plot)
        fig.add_trace(go.Bar(
            x=data['date'], 
            y=data['volume'], 
            name=f"{dash_data['selected_ticker']}" + ' Volume',
            marker=dict(color=dash_data['volume']),  # Custom color
            yaxis='y2',
            opacity=0.2,  # 30% opacity
            hovertemplate= f"{dash_data['selected_ticker']}"+' Volume : %{y:.1s}<extra></extra>'
        ))
        fig.update_layout(
            legend=dict(
                x=0.01,  # Position the legend inside the plot area
                y=0.99,
                xanchor='left',
                yanchor='top',
                bgcolor= dash_data['annot_bg_color'],  # Same as plot_bgcolor with controlled opacity
                bordercolor='rgba(0, 0, 0, 0)',  # Transparent border
            ),
            yaxis2=dict(
                #title='Volume',
                overlaying='y',
                side='right',
                showgrid=False,
                showticklabels=False
            ),
        )
        custom_title = "Volatility Trends: 50-Day MA"
        dash_data['selected_tab_bottom'] = 4
    
    elif tab == 5:
        # Beta Plot
        fig.add_trace(go.Scatter(
            x=data['date'], 
            y=data['beta'], 
            mode='lines',
            name=f"{dash_data['selected_ticker']} Beta",
            line=dict(color=dash_data['stock_color'], width = dash_data['line_width'][2]),  # Blue
            hovertemplate=f"{dash_data['selected_ticker']}" + ' : %{y:.2f}%<extra></extra>'
        ))
        # add S&P500 beta and a daily beta plot
        fig.add_trace(go.Scatter(
            x=data['date'], 
            y=data2['beta'], 
            mode='lines',
            name='SPY Beta',
            line=dict(color=dash_data['color1'], width = dash_data['line_width'][3]),  # Blue
            hovertemplate='SPY : %{y:.2f}%<extra></extra>'
        ))
        fig.update_layout(
            legend=dict(
                x=0.01,  # Position the legend inside the plot area
                y=0.99,
                xanchor='left',
                yanchor='top',
                bgcolor= dash_data['annot_bg_color'],  # Same as plot_bgcolor with controlled opacity
                bordercolor='rgba(0, 0, 0, 0)',  # Transparent border
            ),
        )
        custom_title = "Volatility Analysis : 50-Day Rolling Beta"
        dash_data['selected_tab_bottom'] = 5
        
    
    else:
        raise ValueError("Invalid tab value. Please select a valid tab (0-5).")

    # Add custom title annotation inside the plot area
    fig.add_annotation(
        text=(
            f"<span style='color:{dash_data['g2_title_color']}; font-weight:normal; font-variant:small-caps; font-size:0.8vw;'>{custom_title}</span><br><br>"
        ),
        xref="paper", yref="paper",          # Reference the plot area
        x=0.5, y=1,                      # Position (x, y) inside the plot area
        xanchor="center", yanchor="top",
        yshift=20,
        showarrow=False,                     # No arrow pointing to the text
        align="center",                        # Left-align the text
        borderwidth=0,
        borderpad=0,
        bgcolor=dash_data['paper_bg_color'],
        opacity=1
    )

    fig.update_layout(
        yaxis=dict(showticklabels=False, gridcolor=dash_data['plot_grid_color']),
        xaxis=dict(showgrid=False, tickfont=dict(color=dash_data['text_color'])),
        plot_bgcolor=dash_data['plot_bg_color'],
        paper_bgcolor=dash_data['paper_bg_color'],
        hovermode='x unified',
        margin=dict(l=0, r=0, t=20, b=0),
        #showlegend=False,
        modebar_remove=['zoom', 'zoomin', 'zoomout', 'pan', 'select', 'lasso', 'resetScale2d', 'toimage'],
        modebar_add=['autoscale'],
        modebar=dict(
            orientation='h',
            bgcolor='rgba(0, 0, 0, 0)',
            color='lightgrey',
            activecolor='grey'
        )
    )
    return fig
##########--------------------##########--------------------##########--------------------
def get_ticker_infobar(tickers):
    # Function to retrieve and update stock details for the selected ticker
    """
    NAME: get_ticker_infobar
    DESCRIPTION: Retrieve detailed information about a stock based on its ticker from the Stock_Details and Stock_Price tables.
    PARAMETERS:
        ticker (str): The stock ticker.
    RETURNS:
        - None: The function updates the global dash_data dictionary with the retrieved values such as short name, current price, date, delta price, delta percentage, ticker colors, and range levels.
    """
    with sqlite3.connect('NewStockData.db') as conn:
        # Query to get short name from Stock_Details
        query_details = "SELECT short_name FROM Stock_Details WHERE ticker = ?"
        df_details = pd.read_sql(query_details, conn, params=(tickers,))
        
        # Query to get the latest two price details from Stock_Price
        query_price = """
        SELECT date, stock_close, stock_open 
        FROM Stock_Price 
        WHERE stock_id = (SELECT stock_id FROM Stock_Details WHERE ticker = ?)
        ORDER BY date DESC 
        LIMIT 2
        """
        df_price = pd.read_sql(query_price, conn, params=(tickers,))
    
    # If the ticker is not found or there is no price data, set default values
    if df_details.empty or df_price.empty:
        dash_data['short_name'] = None
        dash_data['current_price'] = None
        dash_data['date'] = None
        dash_data['delta_price'] = None
        dash_data['delta_percentage'] = None
        dash_data['ticker_colors'] = None
        for dictionary in dash_data['range_level']:
            dictionary['end'] = None
            dictionary['start'] = None
    else:
        # Extract and store the relevant data in dash_data dictionary
        dash_data['short_name'] = df_details['short_name'].iloc[0]
        dash_data['current_price'] = df_price['stock_close'].iloc[0]
        dash_data['date'] = get_date(df_price['date'].iloc[0])
        for dictionary in dash_data['range_level']:
            dictionary['end'] = df_price['stock_close'].iloc[0]
            dictionary['start'] = df_price['stock_open'].iloc[0]
        
        # Calculate delta price and delta percentage if there are two rows
        if len(df_price) == 2:
            delta_price = df_price['stock_close'].iloc[0] - df_price['stock_close'].iloc[1]
            delta_percentage = (delta_price / df_price['stock_close'].iloc[1]) * 100
            dash_data['delta_price'] = delta_price
            dash_data['delta_percentage'] = delta_percentage
        else:
            dash_data['delta_price'] = None
            dash_data['delta_percentage'] = None
        if delta_percentage < 0:
            dash_data['ticker_colors'] = tick_colors['Negative']
        else:
            dash_data['ticker_colors'] = tick_colors['Positive']
get_ticker_infobar(dash_data['selected_ticker'])
def get_gauge_info(tickers):
    # Function to retrieve and update min, max, and return levels for the gauges
    """
    NAME: get_gauge_info
    DESCRIPTION: Retrieves the minimum, maximum, start, end, and median values for stock close prices over various timeframes and calculates the return levels. 
                 Updates the global dash_data dictionary with these values.
    PARAMETERS:
        - tickers (str): The stock ticker symbol.
    RETURNS:
        - None: The function updates the global dash_data dictionary with range levels and return levels for different timeframes.
    """

    
    current_date = datetime.strptime(dash_data['date'].replace('on ', ''), '%d-%b, %Y').date()
    
    # Create date_limit list with required dates
    date_limit = [
        current_date - timedelta(weeks=1),
        current_date - relativedelta(months=1),
        current_date - relativedelta(years=1),
        current_date - relativedelta(years=5)
    ]
    
    for i, limit_date in enumerate(date_limit):
        with sqlite3.connect('NewStockData.db') as conn:
            # Perform query for min and max stock_close
            query_min = """
            SELECT MIN(stock_close) as min_close, MAX(stock_close) as max_close
            FROM Stock_Price
            WHERE stock_id = (SELECT stock_id FROM Stock_Details WHERE ticker = ?)
            AND date BETWEEN ? AND ?
            """
            # Replace with actual query execution, e.g., using a database cursor
            min_max_values = pd.read_sql(query_min, conn, params=(tickers, limit_date, current_date)).iloc[0]
            min_value = min_max_values['min_close']
            max_value = min_max_values['max_close']
            
            # Save the min and max values to dash_data
            dash_data['range_level'][i] = {
                'min': min_value.round(2), 
                'max': max_value.round(2),
                'start': dash_data['range_level'][i].get('start'),  # Ensure 'start' exists
                'end': dash_data['range_level'][i].get('end'),      # Ensure 'end' exists
                'median': dash_data['range_level'][i].get('median') # Ensure 'median' exists
            }        
            # Query for the stock_close values
            query_close = """
            SELECT stock_close
            FROM Stock_Price
            WHERE stock_id = (SELECT stock_id FROM Stock_Details WHERE ticker = ?)
            AND date BETWEEN ? AND ?
            ORDER BY date ASC
            """
            buffer_date = limit_date + timedelta(weeks=1)
            # Replace with actual query execution, e.g., using a database cursor
            before_value = pd.read_sql(query_close, conn, params=(tickers, limit_date, buffer_date)).iloc[0]['stock_close']
            #dash_data['before_value'][i] = before_value
            # Calculate returns and save to dash_data
            dash_data['return_level'][i]['start'] = get_returns(before_value, dash_data['range_level'][i].get('start'))
            dash_data['return_level'][i]['end'] = get_returns(before_value, dash_data['range_level'][i].get('end'))
            min_return = get_returns(before_value, dash_data['range_level'][i].get('min'))
            max_return = get_returns(before_value, dash_data['range_level'][i].get('max'))
            # Calculate returns for SPY (or any benchmark ticker)
            spy_before_value = pd.read_sql(query_close, conn, params=('SPY', limit_date, buffer_date)).iloc[0]['stock_close']
            #dash_data['spy_value'][i] = spy_before_value
            spy_now_value = pd.read_sql(query_close, conn, params=('SPY', current_date, current_date)).iloc[0]['stock_close']
            spy_return_value = get_returns(spy_before_value, spy_now_value)
            dash_data['return_level'][i]['median'] = spy_return_value

            dash_data['return_level'][i]['min'] = min(min_return, max_return, dash_data['return_level'][i]['start'], dash_data['return_level'][i]['end'], dash_data['return_level'][i]['median'])
            dash_data['return_level'][i]['max'] = max(min_return, max_return, dash_data['return_level'][i]['start'], dash_data['return_level'][i]['end'], dash_data['return_level'][i]['median'])
get_gauge_info(dash_data['selected_ticker'] )
##########--------------------##########--------------------##########--------------------
# Dictionary to store configuration settings for various placeholders in the Dash application
"""
    NAME: placeholder_config (Dictionary)
    DESCRIPTION: A configuration dictionary that stores settings and components for various placeholders in the Dash application, including ticker infobars, search bars, dropdowns, gauges, and graphs.
    PARAMETERS:
        - 'ticker_infobar': (dict) Configuration for the ticker infobar placeholder.
            - 'indicator': (str): A placeholder for an indicator element, which could be configured with further details.
        
        - 'ticker_searchbar': (dict) Configuration for the ticker search bar placeholder.
            - 'backward_indicator': (html.Button): A button for navigating to the previous ticker.
            - 'ticker_select': (dcc.Dropdown): A dropdown component for selecting a ticker.
            - 'forward_indicator': (html.Button): A button for navigating to the next ticker.

        - 'category_dropdown': (dict) Configuration for the category dropdown placeholder.
            - 'category_label': (str): A label for the category dropdown (currently empty, can be customized).
            - 'dropdown_element': (dcc.Dropdown): The actual dropdown component for selecting stock categories.

        - 'gauges': (dict) Configuration for the gauge placeholders.
            - 'gauge_titles': (list of str): Titles for each gauge, e.g., '1-Week', '1-Month', '1-Year', '5-Year'.
            - 'range_label': (str): A label describing the range displayed by the gauges (e.g., "Current Open to Close Price (in $)").
            - 'return_label': (str): A label describing the returns displayed by the gauges (e.g., "Current Open to Close Returns (in %)").
            - 'left_location': (list of str): A list of positions for the left location of each gauge, used for positioning in the layout.
            - 'color': (str): The background color used in the gauge placeholders.
            - 'text': (str): The text color used in the gauge placeholders.

        - 'graphs': (dict) Configuration for the graph placeholders.
            - 'history_plot': (str): A placeholder for the history plot configuration.
            - 'volume_plot': (str): A placeholder for the volume plot configuration.
            - 'trend_analysis': (str): A placeholder for the trend analysis plot configuration.
            - 'volatility_analysis': (str): A placeholder for the volatility analysis plot configuration.
    RETURNS:
        - None: This dictionary is used for configuring UI components throughout the application.
"""
placeholder_config = {
    'ticker_infobar': {
        'indicator': '' #Indicator Dict
    },
    'ticker_searchbar': {
        'backward_indicator': create_left_button(),
        'ticker_select': create_ticker_dropdown(),
        'forward_indicator': create_right_button()
    },
    'category_dropdown': {
        'category_label': '',
        'dropdown_element': create_category_dropdown()  # Storing the dropdown component
    },
    'gauges': {
        'gauge_titles': ['1-Week', '1-Month', '1-Year', '5-Year'],
        'range_label': 'Current Open to Close Price (in $)',
        'return_label': 'Current Open to Close Returns (in %)',
        'left_location': ['2.875vw', '27.625vw', '52.375vw', '77.125vw'],
        'color': 'white',
        'text': 'black'
    },
    'graphs': {
        'history_plot': 'History Plot Dict',
        'volume_plot': 'Volume Plot Dict',
        'trend_analysis': 'Trend Analysis Dict',
        'volatility_analysis': 'Volatility Analysis Dict'
    }
}
##########--------------------##########--------------------##########--------------------
def create_range_slider_container(config, ID):
    # Function to create a range slider container with a custom median marker
    """
    NAME: create_range_slider_container
    DESCRIPTION: Creates a container for a range slider component, which includes a range slider, an overlay, and an optional marker for a median value.
    PARAMETERS:
        - config (dict): A dictionary containing configuration values such as min, max, start, end, and median.
        - ID (str): A unique identifier for the range slider component.
    RETURNS:
        - html.Div: A Dash HTML Div element containing the range slider and other components.
    """
    # Extract values from the config dictionary
    '''
    min = config.get('min', 0)
    max = config.get('max', 1)
    start = config.get('start', 0)
    end = config.get('end', 1)
    median = config.get('median', (min + max) / 2)
    '''
    min = config['min']
    max = config['max']
    start = config['start']
    end = config['end']
    median = config.get('median')

    # Calculate the position of the important_median as a percentage relative to the total range
    if median is not None:
        important_median_position = 100 * (median - min) / (max - min)

        # Green triangle marker div
        important_median_marker_div = html.Div(
            style={
                'position': 'absolute',
                'top': '0vh',
                'left': f'{important_median_position}%',
                'transform': 'translateX(-50%)',
                'width': '0',
                'height': '0',
                'borderLeft': '1vh solid transparent',
                'borderRight': '1vh solid transparent',
                'borderTop': '1vh solid green',
                'zIndex': '2'
            },
            children=[
                html.Div(
                    f'SPY500: {median}',
                    className='custom-tooltip',
                    style={'visibility': 'hidden'}
                )
            ],
            className='marker-container'
        )
    else:
        important_median_marker_div = None

    # Grey overlay div
    overlay_div = html.Div(
        style={
            'backgroundColor': 'transparent',
            'position': 'absolute',
            'top': '0',
            'left': '0',
            'height': '100%',
            'width': '100%',
            'zIndex': '1'
        }
    )

    # RangeSlider component div
    range_slider_div = html.Div(
        dcc.RangeSlider(
            id=f'range-slider-{ID}',  # Unique ID for each slider
            min=min,
            max=max,
            value=[start, end],
            marks={
                min: {'label': str(round(min)), 'style': {'color': placeholder_config['gauges']['text'], 'transform': 'translateX(0.15vw)','top': '-1vh', 'font-weight': 'bold', 'font-size': '1vw', 'textAlign': 'left', 'border': '0px solid black'}},
                max: {'label': str(round(max)), 'style': {'color': placeholder_config['gauges']['text'], 'position': 'absolute','left':'12.25vw', 'top': '-1vh', 'width': '5vw', 'font-weight': 'bold', 'font-size': '1vw', 'textAlign': 'right', 'border': '0px solid black'}},
            },
            tooltip={"placement": "top", "always_visible": False},
            updatemode='drag',
            className=f'custom-slider {"red-slider" if end < start else "green-slider"}'  # Two classes: custom-slider and red/green-slider
        ),
        style={
            'position': 'absolute',
            'top': '0.5vh',
            'padding': '0',
            'margin': '0',
            'width': '100%',
            'boxSizing': 'border-box'
        }
    )

    # Assemble children divs
    children = [range_slider_div, overlay_div]
    if important_median_marker_div:
        children.insert(1, important_median_marker_div)

    # Flatten the children list if there are no extra levels of nesting
    return html.Div(children=children, style={
        'height': '100%', 
        'width': '100%', 
        'position': 'relative'
    })
def create_gauge_placeholder(index):    
    # Function to create a gauge placeholder with range and return levels
    """
    NAME: create_gauge_placeholder
    DESCRIPTION: Creates a placeholder for a gauge, including the gauge title, range slider, and return slider components.
    PARAMETERS:
        - index (int): The index of the gauge, which determines its position and associated data in the dash_data dictionary.
    RETURNS:
        - html.Div: A Dash HTML Div element structured to display the gauge title, range level, and return level components.
    """
    gauge_title_placeholder = html.Div(placeholder_config['gauges']['gauge_titles'][index], style={'background-color': placeholder_config['gauges']['color'], 'height': '4vh', 'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center',  'color' : 'black', 'color':dash_data['text_color'], 'font-variant': 'small-caps'})
    #gauge_range_label_placeholder = html.Div(placeholder_config['gauges']['range_label'], style={'background-color': placeholder_config['gauges']['color'], 'height': '8vh', 'width': '5vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'color' : 'black'})
    gauge_range_level_placeholder = html.Div(id={'type': 'gauge-range-level', 'index': index}, children = create_range_slider_container(dash_data['range_level'][index], f"NGE-{index}"), style={'position': 'relative','background-color': placeholder_config['gauges']['color'], 'height': '8vh', 'width': '15vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'color' : 'black'})    
    #gauge_returns_label_placeholder = html.Div(placeholder_config['gauges']['return_label'], style={'background-color': placeholder_config['gauges']['color'], 'height': '8vh', 'width': '5vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center',  'color' : 'black'})
    gauge_returns_level_placeholder = html.Div(id={'type': 'gauge-return-level', 'index': index}, children = create_range_slider_container(dash_data['return_level'][index], f"URN-{index}"), style={'position': 'relative','background-color': placeholder_config['gauges']['color'], 'height': '8vh', 'width': '15vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'color' : 'black'})
    
    gauge_placeholder = html.Div(
        [
            gauge_title_placeholder,
            html.Div(
                [
                    #gauge_range_label_placeholder,
                    gauge_range_level_placeholder
                ],
                style={'display': 'flex', 'width': '100%'}
            ),
            html.Div(
                [
                    #gauge_returns_label_placeholder,
                    gauge_returns_level_placeholder
                ],
                style={'display': 'flex', 'width': '100%'}
            )
        ],
        #style={'background-color': 'Salmon', 'height': '20vh', 'width': '15vw', 'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'justify-content': 'center', 'position': 'absolute', 'left':placeholder_config['gauges']['left_location'][index]}
        style={'background-color': 'Salmon', 'height': '20vh', 'width': '15vw', 'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'justify-content': 'center'}
    )
    
    return gauge_placeholder
##########--------------------##########--------------------##########--------------------
def update_category(cat=None):
    # Function to update the selected category and associated ticker list
    """
    NAME: update_category
    DESCRIPTION: Updates the selected category and ticker list based on the provided category. If no category is provided, it resets to show all tickers.
    PARAMETERS:
        - cat (str or None, optional): The category name to filter tickers by. If None, shows all tickers. Default is None.
    RETURNS:
        - list: The updated list of tickers.
        - str: The first ticker from the updated list, set as the selected ticker.
    """
    if cat is None:
        dash_data['selected_category'] = None
        dash_data['ticker_list'] = get_tickers()
        dash_data['selected_ticker'] = dash_data['ticker_list'][0]
    else:
        dash_data['selected_category'] = cat
        dash_data['ticker_list'] = get_tickers(cat)
        dash_data['selected_ticker'] = dash_data['ticker_list'][0]
    
    return dash_data['ticker_list'], dash_data['selected_ticker']
def update_ticker(left=None, right=None):
    # Function to update the selected ticker based on left or right navigation
    """
    NAME: update_ticker
    DESCRIPTION: Updates the selected ticker based on left or right navigation. The selection cycles through the ticker list.
    PARAMETERS:
        - left (bool or None, optional): If True, select the previous ticker. Default is None.
        - right (bool or None, optional): If True, select the next ticker. Default is None.
    RETURNS:
        - str: The updated selected ticker.
    """
    if left is None and right is None:
        return dash_data['selected_ticker']
    
    current_index = dash_data['ticker_list'].index(dash_data['selected_ticker'])
    
    if left is not None:
        new_index = (current_index - 1) % len(dash_data['ticker_list'])
    elif right is not None:
        new_index = (current_index + 1) % len(dash_data['ticker_list'])
    
    dash_data['selected_ticker'] = dash_data['ticker_list'][new_index]
    return dash_data['selected_ticker']
def update_ticker_infobar(tick=None):
    # Function to update the ticker infobar with detailed stock information
    """
    NAME: update_ticker_infobar
    DESCRIPTION: Updates the ticker infobar with detailed stock information, including short name, current price, date, delta price, delta percentage, and ticker colors.
    PARAMETERS:
        - tick (str or None, optional): The ticker symbol for which to update the infobar. If None, uses the currently selected ticker in dash_data. Default is None.
    RETURNS:
        - tuple: A tuple containing the short name, current price (formatted), date, delta price (formatted), delta percentage (formatted), and ticker colors.
    """
    if tick is None:
        tick = dash_data['selected_ticker']
    
    get_ticker_infobar(tick)
    
    return dash_data['short_name'], f"${dash_data['current_price'].round(2)}", dash_data['date'], f"${dash_data['delta_price'].round(1)}", f"({dash_data['delta_percentage'].round(1)})%", dash_data['ticker_colors']
def update_gauge_info(tick=None):
    # Function to update gauge information for the selected ticker
    """
    NAME: update_gauge_info
    DESCRIPTION: Updates the gauge information, including range levels and return levels for the selected ticker.
    PARAMETERS:
        - tick (str or None, optional): The ticker symbol for which to update the gauge information. If None, uses the currently selected ticker in dash_data. Default is None.
    RETURNS:
        - list: The updated range levels for the selected ticker.
        - list: The updated return levels for the selected ticker.
    """
    if tick is None:
        tick = dash_data['selected_ticker']
    
    get_gauge_info(tick)
    
    return dash_data['range_level'], dash_data['return_level']
def update_plots (tick = None):
    # Function to update all plots based on the selected ticker
    """
    NAME: update_plots
    DESCRIPTION: Updates the time series, volume, trend, and volatility plots based on the selected ticker.
    PARAMETERS:
        - tick (str or None, optional): The ticker symbol for which to update the plots. If None, uses the currently selected ticker in dash_data. Default is None.
    RETURNS:
        - tuple: A tuple containing the updated figures for time series plot, volume plot, trend plot, and volatility plot.
    """
    if tick is None:
        tick = dash_data['selected_ticker']
    
    dash_data['time_series_data'] = get_stock_data(tick)
    dash_data['time_series_data']['date'] =  pd.to_datetime(dash_data['time_series_data']['date'])
    dash_data['end_date'] = dash_data['time_series_data']['date'].max()
    dash_data['start_date'] = dash_data['end_date'] - pd.DateOffset(months=3)
    dash_data['g2_start_date'] = dash_data['start_date']
    filtered_data = dash_data['time_series_data'][(dash_data['time_series_data']['date'] >= dash_data['start_date']) & (dash_data['time_series_data']['date'] <= dash_data['end_date'])]
    filtered_data2 = dash_data['SPY_data'][(dash_data['SPY_data']['date'] >= dash_data['start_date']) & (dash_data['SPY_data']['date'] <= dash_data['end_date'])]
    figure1 =create_time_series_plot(filtered_data.drop(columns=['volume_50_day_avg']),dash_data['selected_columns_line'],'line')
    figure2 =create_volume_plot(filtered_data[['date','volume', 'volume_50_day_avg']])
    figure3 =create_trend_plot(filtered_data,dash_data['selected_tab_top'])
    figure4=create_volatility_plot(filtered_data,dash_data['selected_tab_bottom'],filtered_data2)
    return figure1, figure2, figure3, figure4
##########--------------------##########--------------------##########--------------------
#Dashboard Layout
## Header components
title_placeholder = html.H1(dash_data['header'], style={'color': 'darkcyan','background-color': 'white', 'height': '5vh', 'margin': '0vh 1vw', 'font-variant': 'small-caps', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'flex-grow': 1, 'white-space': 'nowrap'})
header_placeholder = html.Div(
    title_placeholder,
    style={
        'background-color': 'white', 
        'height': '5vh', 
        'width': '99vw', 
        'margin': '0.01vh 1vw', 
        'display': 'flex', 
        'align-items': 'center', 
        'justify-content': 'center'
    }
)
## Selection components
# Top left short name and ticker
short_name_placeholder = html.Div(id='short-name',children = dash_data['short_name'], style={'height': '3vh', 'flex-grow': 0, 'flex-shrink': 1, 'flex-basis': 'auto', 'display': 'flex', 'align-items': 'center', 'justify-content': 'flex-start', 'padding': '0 5px', 'overflow': 'hidden', 'text-overflow': 'ellipsis', 'white-space': 'nowrap', 'font-variant': 'small-caps', 'font-weight': 'bold', 'color': 'darkcyan', 'font-size': '2vh', 'font-family': 'Linotype Aroma, Arial, sans-serif'})
ticker2_placeholder = html.Div(id='selected-ticker',children = f"( {dash_data['selected_ticker']} )", style={'height': '3vh', 'flex-grow': 0, 'flex-shrink': 1, 'flex-basis': 'auto', 'display': 'flex', 'align-items': 'center', 'justify-content': 'flex-start','padding': '0 5px', 'overflow': 'hidden', 'text-overflow': 'ellipsis', 'white-space': 'nowrap', 'font-variant': 'small-caps', 'color': 'darkcyan', 'font-size': '2vh', 'font-family': 'Linotype Aroma, Arial, sans-serif'})
topleft_placeholder = html.Div(
    [
        short_name_placeholder,
        ticker2_placeholder
    ],
    style={'display': 'flex', 'width': '20vw'}
)
# Bottom left indicator and price
indicator_placeholder = html.Div(placeholder_config['ticker_infobar']['indicator'], style={'background-color': 'transparent', 'height': '7vh', 'width': '2vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'border': '0px solid black'}) #'width': '10vw'
price_placeholder = html.Div(id='current-price', children = f"${dash_data['current_price'].round(2)}", style={'height': '7vh', 'width': '10vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'color': 'black', 'font-size': '6vh', 'font-family': 'Linotype Aroma, Arial, sans-serif', 'font-weight': 'bold'})
bottomleft_placeholder = html.Div(
    [
        indicator_placeholder,
        price_placeholder
    ],
    style={'display': 'flex', 'width': '20vw'}
)
## Left side container
left_side_placeholder = html.Div(
    [
        topleft_placeholder,
        bottomleft_placeholder
    ],
    style={'display': 'flex', 'flex-direction': 'column', 'width': '20vw'}
)
# Right side split into three sections
right_top_placeholder = html.Div(id='date',children = dash_data['date'], style={'height': '2.5vh', 'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'color': 'darkgrey', 'font-variant': 'small-caps', 'color':dash_data['text_color'] })
right_middle_placeholder = html.Div(id='delta-price', children = f"${dash_data['delta_price'].round(1)}", style={'height': '3.6vh', 'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'color': dash_data['ticker_colors'], 'font-size': '3vh', 'font-family': 'Linotype Aroma, Arial, sans-serif'})
right_bottom_placeholder = html.Div(id='delta-percentage', children = f"({dash_data['delta_percentage'].round(1)}%)", style={'height': '3.6vh', 'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'color': dash_data['ticker_colors'], 'font-size': '2vh', 'font-family': 'Linotype Aroma, Arial, sans-serif'})
right_side_placeholder = html.Div(
    [
        right_top_placeholder,
        right_middle_placeholder,
        right_bottom_placeholder
    ],
    style={'display': 'flex', 'flex-direction': 'column', 'width': '10vw'}
)
## Main ticker infobar placeholder
ticker_infobar_placeholder = html.Div(
    [
        left_side_placeholder,
        right_side_placeholder
    ],
    style={'height': '10vh', 'width': '30vw', 'display': 'flex', 'position': 'absolute', 'left': '11.5vw'}
)
## Ticker searchbar components
backward_indicator_placeholder = html.Div(placeholder_config['ticker_searchbar']['backward_indicator'], style={'background-color': 'skyblue', 'height': '5vh', 'width': '4vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})
ticker_select_placeholder = html.Div(placeholder_config['ticker_searchbar']['ticker_select'], style={'background-color': 'skyblue', 'height': '5vh', 'width': 'calc(100% - 8vw)', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})
forward_indicator_placeholder = html.Div(placeholder_config['ticker_searchbar']['forward_indicator'], style={'background-color': 'skyblue', 'height': '5vh', 'width': '4vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})
ticker_searchbar_placeholder = html.Div(
    [
        backward_indicator_placeholder,
        ticker_select_placeholder,
        forward_indicator_placeholder
    ],
    style={'background-color': 'skyblue', 'height': '5vh', 'width': '16vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'border': '0px solid black', 'position': 'absolute', 'left': '41.5vw', 'top': '2.5vh'}
)
## Category Filter components
#category_dropdown_placeholder = html.Div("Category Dropdown", style={'background-color': 'PowderBlue', 'height': '10vh', 'width': '30vw', 'display': 'flex', 'align-items': 'center', 'justify-content': 'flex-start', 'border': '1px solid black'})
category_label_placeholder = html.Div(placeholder_config['category_dropdown']['category_label'], style={'color': 'grey','background-color': 'white', 'height': '2.5vh', 'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'flex-start'})
category_dropdown_element_placeholder = html.Div(placeholder_config['category_dropdown']['dropdown_element'], style={'background-color': 'white', 'height': '5vh', 'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'flex-start'})
category_dropdown_placeholder = html.Div(
    [
        category_label_placeholder,
        category_dropdown_element_placeholder
    ],
    style={'background-color': 'white', 'height': '10vh', 'width': '20vw', 'display': 'flex', 'flex-direction': 'column', 'align-items': 'flex-start', 'justify-content': 'flex-start', 'position': 'absolute', 'left': '57.5vw', 'top': '2.5'}
)

selection_placeholder = html.Div(
    [
        ticker_infobar_placeholder,
        ticker_searchbar_placeholder,
        category_dropdown_placeholder,
    ],
    style={
        'background-color': 'white', 
        'height': '10vh', 
        'width': '99vw', 
        'margin': '0.01vh 1vw', 
        'display': 'flex', 
        'align-items': 'center', 
        'justify-content': 'center',
        'position': 'relative'
        }
)

## Gauge components
gauge_range_label_placeholder = html.Div(placeholder_config['gauges']['range_label'], style={'background-color': placeholder_config['gauges']['color'], 'color':dash_data['text_color'], 'height': '8vh', 'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'border-bottom': '1px dashed #5f6b6d', 'position': 'relative', 'top': '3vh', 'font-variant': 'small-caps', 'color': '#5f6b6d', 'font-size': '2vh', 'font-family': 'Linotype Aroma, Arial, sans-serif'})
gauge_returns_label_placeholder = html.Div(placeholder_config['gauges']['return_label'], style={'background-color': placeholder_config['gauges']['color'], 'color':dash_data['text_color'], 'height': '8vh', 'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'border-top': '0px solid black', 'position': 'relative', 'top': '3vh', 'font-variant': 'small-caps', 'color': '#5f6b6d', 'font-size': '2vh', 'font-family': 'Linotype Aroma, Arial, sans-serif'})
gauge_label_placeholder = html.Div(
    [
        gauge_range_label_placeholder,
        gauge_returns_label_placeholder
    ],
    style={
        'background-color': 'white', 
        'height': '20vh', 
        'width': '10vw', 
        'display': 'flex', 
        'align-items': 'top', 
        'flex-direction': 'column',  # Aligns children in a column
        'justify-content': 'top',
        'border': '0px solid black',
        'position': 'relative',
        'left': '0vw',
        'top': '0vh'
    }
)
# Creating the specified placeholders
gauge1_placeholder = create_gauge_placeholder(0)
gauge2_placeholder = create_gauge_placeholder(1)
gauge3_placeholder = create_gauge_placeholder(2)
gauge4_placeholder = create_gauge_placeholder(3)
gauge_screen_placeholder = html.Div(
    [
        gauge1_placeholder,
        gauge2_placeholder,
        gauge3_placeholder,
        gauge4_placeholder
    ],
    style={
        'background-color': 'white', 
        'height': '20vh', 
        'width': '89vw', 
        'display': 'flex', 
        'align-items': 'center', 
        'justify-content': 'space-around',
        'border': '0px solid black',
        'position': 'relative',
        'left': '0vw',
        'top': '0vh'
    }
)

gauge_placeholder = html.Div(
    [
        gauge_label_placeholder,
        gauge_screen_placeholder
    ],
    style={
        'background-color': 'white', 
        'height': '20vh', 
        'width': '99vw', 
        'margin': '0.01vh 1vw', 
        'display': 'flex', 
        'align-items': 'center', 
        'justify-content': 'center',
        'border': '0px solid black'
    }
)

# Add the buttons for time period selection
button_placeholder = html.Div(
        [
            dbc.Button("1W", id='btn-1W', color="primary", className="custom-button", style={'margin-right': '0.005vw', 'color': 'grey', 'font-size': '0.9vw', 'border': '0px solid black','height': '5vh','width': '14%','display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
            dbc.Button("2W", id='btn-2W', color="primary", className="custom-button", style={'margin-right': '0.005vw', 'color': 'grey', 'font-size': '0.9vw', 'border': '0px solid black','height': '5vh','width': '14%','display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
            dbc.Button("3M", id='btn-3M', color="primary", className="custom-button custom-button-active", style={'margin-right': '0.005vw', 'color': 'grey', 'font-size': '0.9vw', 'border': '0px solid black','height': '5vh','width': '14%','display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
            dbc.Button("6M", id='btn-6M', color="primary", className="custom-button", style={'margin-right': '0.005vw', 'color': 'grey', 'font-size': '0.9vw', 'border': '0px solid black','height': '5vh','width': '14%','display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
            dbc.Button("1Y", id='btn-1Y', color="primary", className="custom-button", style={'margin-right': '0.005vw', 'color': 'grey', 'font-size': '0.9vw', 'border': '0px solid black','height': '5vh','width': '14%','display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
            dbc.Button("5Y", id='btn-5Y', color="primary", className="custom-button", style={'margin-right': '0.005vw', 'color': 'grey', 'font-size': '0.9vw', 'border': '0px solid black','height': '5vh','width': '14%','display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
            dbc.Button("From 2000", id='btn-All', color="primary", className="custom-button", style={'margin-right': '0.005vw', 'color': 'grey', 'font-size': '0.9vw', 'border': '0px solid black','height': '5vh','width': '16%','display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
        ],
        style={
            'background-color': 'transparent', 
            'height': '5vh', 
            'width': '35%', 
            'margin': '0.001vh 0.1vw', 
            'display': 'flex',
            'align-items': 'center', 
            'justify-content': 'left',
            'border-right': '1px dashed black',
            'position': 'relative',
            'left':'0vw'
        }
)
toggle_placeholder = html.Div(
    [
        # Checklist element for short/long term options
        html.Div(
            dbc.Checklist(
                id='ema-term-toggle',
                options=[
                    {'label': '10-Day', 'value': 'ema_10_day'},
                    {'label': '20-Day', 'value': 'ema_20_day'},
                    {'label': '50-Day', 'value': 'ema_50_day'},
                    {'label': '200-Day', 'value': 'ema_200_day'}
                ],
                value=dash_data['selected_columns_line'],  # Default selection
                inline=True,
                style={'font-size': '0.9vw', 'color': 'grey',
                    'display': 'flex',  # Ensures inline display
                    'width': '100%',
                    #'flex-direction': 'row',  # Forces horizontal layout
                    'flex-wrap': 'wrap',  # Allows wrapping if necessary
                    'border': '0px solid black', 'align-items': 'center', 'justify-content': 'center'
                    }
            ),
            style={
                'display': 'flex',
                'flex-direction': 'row',
                'justify-content': 'center',  # Centers the checklist horizontally
                'border-right': '1px dashed black',
                'align-items': 'center',      # Centers the checklist vertically (if necessary)
                'height': '100%',             # Ensures the div takes full height
                'width': '80%'               # Ensures the div takes full width
            }
        ),

        # Dropdown for selecting plot type
        dcc.Dropdown(
            id='plot-type-dropdown',
            options=[
                {'label': 'Line Plot', 'value': 'line'},
                {'label': 'Candlestick', 'value': 'candlestick'},
                {'label': 'Mountain', 'value': 'mountain'}
            ],
            value='line',  # Default selection
            clearable=False,
            style={
                'margin-left': '0.1vw',
                'width': '8vw',
                'font-size': '0.9vw',
                'color': 'grey',
                'position': 'relative',
                'border': '0px solid black',
                'left': '0vw'
            }
        )
    ],
    style={
        'background-color': 'transparent',
        'height': '5vh',
        'width': '70%',
        'margin': '0.001vh 0.1vw',
        'display': 'flex',
        'flex-direction': 'row',
        'align-items': 'center',
        'justify-content': 'space-around',
        'border': '0px solid black',
        'position': 'relative',
        'left': '0vw'
    }
)
filtered_data = dash_data['time_series_data'][(dash_data['time_series_data']['date'] >= dash_data['start_date']) & (dash_data['time_series_data']['date'] <= dash_data['end_date'])]
filtered_data2 = dash_data['SPY_data'][(dash_data['SPY_data']['date'] >= dash_data['start_date']) & (dash_data['SPY_data']['date'] <= dash_data['end_date'])]
## Graph components
#Left Side Plots components
history_plot_placeholder = dcc.Graph(id='time-series-plot', figure=create_time_series_plot(filtered_data.drop(columns=['volume_50_day_avg']),dash_data['selected_columns_line'],'line'), style={'background-color': 'white', 'height': '42.5vh', 'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'border-bottom': '0px dashed grey', 'position': 'relative', 'top':'0vh', 'zIndex': '1'})
volume_plot_placeholder = dcc.Graph(id='volume-plot', figure=create_volume_plot(filtered_data[['date','volume', 'volume_50_day_avg']]), style={'background-color': 'white', 'height': '12.5vh', 'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'border': '0px solid black', 'position': 'relative', 'top':'0vh', 'zIndex': '1'})
range_slider_placeholder = html.Div([button_placeholder, toggle_placeholder], style={'background-color': 'transparent', 'height': '5vh', 'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'border': '0px solid black', 'position': 'relative', 'top': '0vh', 'left': '0vw', 'zIndex': '2'})
graph1_placeholder = html.Div(
        [
            range_slider_placeholder,
            history_plot_placeholder,
            volume_plot_placeholder
        ],
        style={'background-color': 'white', 'height': '60vh', 'width': '48vw', 'display': 'flex', 'flex-direction': 'column', 'margin': '0vh 0.5vw', 'align-items': 'center', 'justify-content': 'center', 'border': '0px solid black', 'position': 'relative',}
        )

#Right Side Plot components
period_dropdown_element = dcc.Dropdown(
        id='period-dropdown',
        options=[
            {'label': '1W', 'value': '1W'},
            {'label': '2W', 'value': '2W'},
            {'label': '3M', 'value': '3M'},
            {'label': '6M', 'value': '6M'},
            {'label': '1Y', 'value': '1Y'},
            {'label': '5Y', 'value': '5Y'},
            {'label': 'From 2000', 'value': 'All'}
        ],
        value='3M',  # Default value
        clearable=False,
        style={
            'width': '7vw',
            'height': '100%',
            'position': 'absolute',
            'left': '0vw',  # Position it to the left
            'top': '0vh',  # Vertically center the dropdown relative to its parent
            'border': '0px solid black',
            'font-size': '0.9vw',  # Adjusted font size
            'padding': '0',  # Remove padding
            'margin': '0',  # Remove margin
            #'transform': 'translateY(-50%)',  # Adjust for perfect centering
            'z-index': '10'  # Ensure it is on top if overlapping with other elements
        },
        className="period-dropdown"
    )
trend_tab_slider = html.Div(
    [
        dcc.Slider(
            min=0,
            max=5,
            step=1,
            value=0,
            marks={i: '' for i in range(6)},  # Empty marks at each integer value
            #tooltip={"placement": "top", "always_visible": False},
            id='trend-tab-slider',
            className="custom-slider-trial4"
        )
    ],
    style={
        'height': '100%',
        'width': '25%',
        'border': '0px solid black',
        'background-color': 'transparent',
        'position': 'absolute',  # Allow positioning with top
        'top': '1.5vh',            # Move the slider down by 50% of the container's height
        'transform': 'translateY(-50%)'  # Center the slider vertically
    }
)
trend_selection_placeholder = html.Div([period_dropdown_element, trend_tab_slider],
    id='trend-selection-placeholder',
    style={
        'height': '10%',
        'width': '100%',
        'display': 'flex',
        'align-items': 'center',
        'justify-content': 'center',
        'background-color': 'transparent',
        'position': 'relative',  # Allow positioning with top
        'border': '0px solid black'
    }
    )
trend_plot_placeholder = dcc.Graph(id='trend-plot', figure=create_trend_plot(filtered_data,0), style={'background-color': 'lightgreen', 'height': '90%', 'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'border-bottom': '1px dashed lightgrey'})
# Combine the placeholders within the parent container
trend_analysis_placeholder = html.Div(
    [
        trend_selection_placeholder,
        trend_plot_placeholder
    ],
    style={
        'background-color': 'transparent',
        'height': '30vh',
        'width': '100%',
        'border': '0px solid black',
        'display': 'flex',
        'flex-direction': 'column',
        'align-items': 'center',
        'justify-content': 'center'
    }
    )
    
volatility_tab_slider = html.Div(
    [
        dcc.Slider(
            min=0,
            max=5,
            step=1,
            value=0,
            marks={i: '' for i in range(6)},  # Empty marks at each integer value
            #tooltip={"placement": "top", "always_visible": False},
            id='volatility-tab-slider',
            className="custom-slider-trial4"
        )
    ],
    style={
        'height': '100%',
        'width': '25%',
        'border': '0px solid black',
        'background-color': 'transparent',
        'position': 'absolute',  # Allow positioning with top
        'top': '1.5vh',            # Move the slider down by 50% of the container's height
        'transform': 'translateY(-50%)'  # Center the slider vertically
    }
)
volatility_selection_placeholder = html.Div([volatility_tab_slider],
    id='volatility-selection-placeholder',
    style={
        'height': '10%',
        'width': '100%',
        'display': 'flex',
        'align-items': 'center',
        'justify-content': 'center',
        'background-color': 'transparent',
        'position': 'relative',  # Allow positioning with top
        'border': '0px solid black'
    }
    )
volatility_plot_placeholder = dcc.Graph(id='volatility-plot', figure=create_volatility_plot(filtered_data,0,filtered_data2), style={'background-color': 'transparent', 'height': '90%', 'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'border-bottom': '0px dashed grey'})
volatility_analysis_placeholder = html.Div([volatility_selection_placeholder,volatility_plot_placeholder], style={'background-color': 'transparent', 'height': '30vh', 'width': '100%', 'display': 'flex', 'flex-direction': 'column','align-items': 'center', 'justify-content': 'center', 'border': '0px solid red'})
graph2_placeholder = html.Div(
        [
        trend_analysis_placeholder,
        volatility_analysis_placeholder
        ],
        style={'background-color': 'transparent', 'height': '60vh', 'width': '48vw', 'margin': '0vh 0.5vw', 'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'justify-content': 'center', 'border': '0px solid black'}
        )
    
graph_placeholder = html.Div(
        [
        graph1_placeholder,
        graph2_placeholder
        ],
        style={
            #'background-color': 'lightgreen', 
            'background-color': 'transparent', 
            'height': '60vh', 
            'width': '99vw', 
            'margin': '0.01vh 1vw', 
            'display': 'flex', 
            'align-items': 'center', 
            'justify-content': 'center',
            'border': '0px solid black'
            }
            )
##########--------------------##########--------------------##########--------------------
# Design the App Layout
app.layout = html.Div([
    dbc.Row([header_placeholder], style={'margin-bottom': '1vh'}),
    dbc.Row([selection_placeholder], style={'margin-bottom': '1vh'}),
    dbc.Row([gauge_placeholder], style={'margin-bottom': '1vh'}),
    dbc.Row([graph_placeholder], style={'margin-bottom': '1vh'})
],
    style={'width': '100vw', 'height': '100vh', 'padding': '0.5vw'}
)
##########--------------------##########--------------------##########--------------------
# Callbacks
@app.callback(
    [Output('ticker-dropdown', 'options'),
     Output('ticker-dropdown', 'value'),
     Output('short-name', 'children'),
     Output('selected-ticker', 'children'),
     Output('current-price', 'children'),
     Output('date', 'children'),
     Output('delta-price', 'children'),
     Output('delta-percentage', 'children'),
     Output('delta-price', 'style'),
     Output('delta-percentage', 'style'),
     Output({'type': 'gauge-range-level', 'index': 0}, 'children'),
     Output({'type': 'gauge-return-level', 'index': 0}, 'children'),
     Output({'type': 'gauge-range-level', 'index': 1}, 'children'),
     Output({'type': 'gauge-return-level', 'index': 1}, 'children'),
     Output({'type': 'gauge-range-level', 'index': 2}, 'children'),
     Output({'type': 'gauge-return-level', 'index': 2}, 'children'),
     Output({'type': 'gauge-range-level', 'index': 3}, 'children'),
     Output({'type': 'gauge-return-level', 'index': 3}, 'children'),
     dash.dependencies.Output('time-series-plot', 'figure', allow_duplicate=True),
     dash.dependencies.Output('volume-plot', 'figure', allow_duplicate=True),
     dash.dependencies.Output('trend-plot', 'figure', allow_duplicate=True),
     dash.dependencies.Output('volatility-plot', 'figure', allow_duplicate=True)],
    [Input('category-dropdown', 'value'),
     Input('left-button', 'n_clicks'),
     Input('right-button', 'n_clicks'),
     Input('ticker-dropdown', 'value')],
    prevent_initial_call=True
)
def update_tickers_and_selection(category, left_clicks, right_clicks, ticker_dropdown_value):
    # Callback to update ticker selection, infobar, gauges, and plots based on user interactions
    """
    NAME: update_tickers_and_selection
    DESCRIPTION: Updates the options and value of the ticker dropdown, infobar details (short name, current price, date, etc.), gauge levels, and various plots 
                 (time series, volume, trend, volatility) based on user interactions with the category dropdown, ticker navigation buttons, or ticker dropdown selection.
    PARAMETERS:
        - category (str or None): The selected stock category from the dropdown.
        - left_clicks (int or None): The number of clicks on the left navigation button.
        - right_clicks (int or None): The number of clicks on the right navigation button.
        - ticker_dropdown_value (str or None): The selected ticker from the ticker dropdown.
    RETURNS:
        - list: A list of updates for the following components:
            - Ticker dropdown options and selected value.
            - Infobar details including short name, current price, date, delta price, and delta percentage.
            - Styles for delta price and delta percentage.
            - Gauge range and return levels for four different timeframes.
            - Updated figures for the time series plot, volume plot, trend plot, and volatility plot.
    """
    ctx = dash.callback_context

    # If no triggers, return no updates
    if not ctx.triggered:
        return [dash.no_update] * 22

    # Identify which input triggered the callback
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    # Common logic for setting delta price and percentage style
    def set_styles(ticker_color_select):
        delta_price_style = {'height': '3.6vh', 'width': '100%', 'display': 'flex', 
                             'align-items': 'center', 'justify-content': 'center', 
                             'font-size': '3vh', 'font-family': 'Linotype Aroma, Arial, sans-serif'}
        delta_price_style['color'] = ticker_color_select  # Only update the color

        delta_percentage_style = {'height': '3.6vh', 'width': '100%', 'display': 'flex', 
                                  'align-items': 'center', 'justify-content': 'center', 
                                  'font-size': '2vh', 'font-family': 'Linotype Aroma, Arial, sans-serif'}
        delta_percentage_style['color'] = ticker_color_select  # Only update the color
        
        return delta_price_style, delta_percentage_style

    # Handle ticker-dropdown value changes
    if trigger == 'ticker-dropdown':
        selected_ticker = ticker_dropdown_value
        dash_data['selected_ticker'] = selected_ticker
        selected_ticker2 = f"( {selected_ticker} )"
        short_name, current_price, date, delta_price, delta_percentage, ticker_color_select = update_ticker_infobar(selected_ticker)
        delta_price_style, delta_percentage_style = set_styles(ticker_color_select)
        range_level, return_level = update_gauge_info(selected_ticker)
        fig1, fig2, fig3, fig4 = update_plots (tick = dash_data['selected_ticker'])

        return (dash.no_update, selected_ticker, short_name, selected_ticker2, current_price, date, delta_price, delta_percentage, delta_price_style, delta_percentage_style, 
                create_range_slider_container(range_level[0],"NGE-0"), create_range_slider_container(return_level[0],"URN-0"),
                create_range_slider_container(range_level[1],"NGE-1"), create_range_slider_container(return_level[1],"URN-1"),
                create_range_slider_container(range_level[2],"NGE-2"), create_range_slider_container(return_level[2],"URN-2"),
                create_range_slider_container(range_level[3],"NGE-3"), create_range_slider_container(return_level[3],"URN-3"),
                fig1, fig2, fig3, fig4)

    # Handle category dropdown changes
    if trigger == 'category-dropdown':
        ticker_list, selected_ticker = update_category(category)
        
        short_name, current_price, date, delta_price, delta_percentage, ticker_color_select = update_ticker_infobar(selected_ticker)
        selected_ticker2 = f"( {selected_ticker} )"
        options = [{'label': ticker, 'value': ticker} for ticker in ticker_list]
        delta_price_style, delta_percentage_style = set_styles(ticker_color_select)
        # Update gauge info
        range_level, return_level = update_gauge_info(selected_ticker)
        fig1, fig2, fig3, fig4 = update_plots (tick = dash_data['selected_ticker'])

        return (options, selected_ticker, short_name, selected_ticker2, current_price, date, delta_price, delta_percentage, delta_price_style, delta_percentage_style, 
                create_range_slider_container(range_level[0],"NGE-0"), create_range_slider_container(return_level[0],"URN-0"),
                create_range_slider_container(range_level[1],"NGE-1"), create_range_slider_container(return_level[1],"URN-1"),
                create_range_slider_container(range_level[2],"NGE-2"), create_range_slider_container(return_level[2],"URN-2"),
                create_range_slider_container(range_level[3],"NGE-3"), create_range_slider_container(return_level[3],"URN-3"),
                fig1, fig2, fig3, fig4)

    # Handle button clicks
    elif trigger in ['left-button', 'right-button']:
        left = 1 if trigger == 'left-button' else None
        right = 1 if trigger == 'right-button' else None
        #left = 1 if left_clicks else None
        #right = 1 if right_clicks else None

        selected_ticker = update_ticker(left=left, right=right)
        selected_ticker2 = f"( {selected_ticker} )"
        short_name, current_price, date, delta_price, delta_percentage, ticker_color_select = update_ticker_infobar(selected_ticker)
        delta_price_style, delta_percentage_style = set_styles(ticker_color_select)
        # Update gauge info
        range_level, return_level = update_gauge_info(selected_ticker)
        fig1, fig2, fig3, fig4 = update_plots (tick = dash_data['selected_ticker'])

        return (dash.no_update, selected_ticker, short_name, selected_ticker2, current_price, date, delta_price, delta_percentage, delta_price_style, delta_percentage_style, 
                create_range_slider_container(range_level[0],"NGE-0"), create_range_slider_container(return_level[0],"URN-0"),
                create_range_slider_container(range_level[1],"NGE-1"), create_range_slider_container(return_level[1],"URN-1"),
                create_range_slider_container(range_level[2],"NGE-2"), create_range_slider_container(return_level[2],"URN-2"),
                create_range_slider_container(range_level[3],"NGE-3"), create_range_slider_container(return_level[3],"URN-3"),
                fig1, fig2, fig3, fig4 )

    # Default return if nothing is matched
    return [dash.no_update] * 22
@app.callback(
    [dash.dependencies.Output('time-series-plot', 'figure', allow_duplicate=True),
     dash.dependencies.Output('volume-plot', 'figure', allow_duplicate=True),
     dash.dependencies.Output('btn-1W', 'className'),
     dash.dependencies.Output('btn-2W', 'className'),
     dash.dependencies.Output('btn-3M', 'className'),
     dash.dependencies.Output('btn-6M', 'className'),
     dash.dependencies.Output('btn-1Y', 'className'),
     dash.dependencies.Output('btn-5Y', 'className'),
     dash.dependencies.Output('btn-All', 'className')],
    [dash.dependencies.Input('btn-1W', 'n_clicks'),
     dash.dependencies.Input('btn-2W', 'n_clicks'),
     dash.dependencies.Input('btn-3M', 'n_clicks'),
     dash.dependencies.Input('btn-6M', 'n_clicks'),
     dash.dependencies.Input('btn-1Y', 'n_clicks'),
     dash.dependencies.Input('btn-5Y', 'n_clicks'),
     dash.dependencies.Input('btn-All', 'n_clicks'),
     dash.dependencies.Input('ema-term-toggle', 'value'),
     dash.dependencies.Input('plot-type-dropdown', 'value')],
    [State('ticker-dropdown', 'value')],
    prevent_initial_call=True

)
def update_graph_based_on_buttons(btn1w, btn2w, btn3m, btn6m, btn1y, btn5y, btn_all, ema_terms, plot_type, selected_ticker):
    # Callback to update time series and volume plots based on time range and EMA term selections
    """
    NAME: update_graph_based_on_buttons
    DESCRIPTION: Updates the time series and volume plots based on the selected time range buttons (1W, 2W, 3M, etc.), EMA term toggle, and plot type selection. 
                 Also updates the active button's CSS class to indicate selection.
    PARAMETERS:
        - btn1w, btn2w, btn3m, btn6m, btn1y, btn5y, btn_all (int or None): Number of clicks for each respective time range button.
        - ema_terms (list of str or None): The selected EMA terms to display on the time series plot.
        - plot_type (str or None): The selected plot type (line, mountain, or candlestick) for the time series plot.
        - selected_ticker (str): The currently selected ticker from the ticker dropdown.
    RETURNS:
        - tuple: A tuple containing:
            - Updated time series plot figure.
            - Updated volume plot figure.
            - Updated CSS class names for each time range button.
    """
    ctx = dash.callback_context
    data = dash_data['time_series_data']
    
    active_button = None
    
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Handle button selections and store the selected range
        if button_id == 'btn-1W' and btn1w:
            dash_data['start_date'] = dash_data['end_date'] - pd.DateOffset(weeks=1)
            dash_data['selected_button'] = 'btn-1W'
        elif button_id == 'btn-2W' and btn2w:
            dash_data['start_date'] = dash_data['end_date'] - pd.DateOffset(weeks=2)
            dash_data['selected_button'] = 'btn-2W'
        elif button_id == 'btn-3M' and btn3m:
            dash_data['start_date'] = dash_data['end_date'] - pd.DateOffset(months=3)
            dash_data['selected_button'] = 'btn-3M'
        elif button_id == 'btn-6M' and btn6m:
            dash_data['start_date'] = dash_data['end_date'] - pd.DateOffset(months=6)
            dash_data['selected_button'] = 'btn-6M'
        elif button_id == 'btn-1Y' and btn1y:
            dash_data['start_date'] = dash_data['end_date'] - pd.DateOffset(years=1)
            dash_data['selected_button'] = 'btn-1Y'
        elif button_id == 'btn-5Y' and btn5y:
            dash_data['start_date'] = dash_data['end_date'] - pd.DateOffset(years=5)
            dash_data['selected_button'] = 'btn-5Y'
        elif button_id == 'btn-All' and btn_all:
            dash_data['start_date'] = data['date'].min()
            dash_data['selected_button'] = 'btn-All'
        
        # Update the EMA columns if they change
        if dash_data['selected_columns_line'] != ema_terms:
            dash_data['selected_columns_line'] = ema_terms

    # If no button is clicked, maintain the previously selected date range    
    active_button = dash_data.get('selected_button', 'btn-3M')

    # Filter the data based on the selected date range
    filtered_data = data[(data['date'] >= dash_data['start_date']) & (data['date'] <= dash_data['end_date'])]
    
    # Create the figures with the updated inputs
    fig1 = create_time_series_plot(filtered_data, columns = dash_data['selected_columns_line'], plot_type=plot_type)
    fig2 = create_volume_plot(filtered_data)
    
    # Set the class for each button
    btn_classes = {button: 'custom-button' for button in ['btn-1W', 'btn-2W', 'btn-3M', 'btn-6M', 'btn-1Y', 'btn-5Y', 'btn-All']}
    if active_button:
        btn_classes[active_button] = 'custom-button custom-button-active'

    return (fig1, fig2, 
            btn_classes['btn-1W'], 
            btn_classes['btn-2W'], 
            btn_classes['btn-3M'], 
            btn_classes['btn-6M'], 
            btn_classes['btn-1Y'], 
            btn_classes['btn-5Y'], 
            btn_classes['btn-All'])
@app.callback(
    [dash.dependencies.Output('trend-plot', 'figure' , allow_duplicate=True),
     dash.dependencies.Output('volatility-plot', 'figure', allow_duplicate=True)],
    [dash.dependencies.Input('period-dropdown', 'value'),
     dash.dependencies.Input('trend-tab-slider', 'value'),
     dash.dependencies.Input('volatility-tab-slider', 'value')],
    [State('ticker-dropdown', 'value')],
    prevent_initial_call=True
)
def update_trend_plot(period_value, trend_tab_index, volatility_tab_index, selected_ticker):
    # Callback to update trend and volatility plots based on selected period and tabs
    """
    NAME: update_trend_plot
    DESCRIPTION: Updates the trend and volatility plots based on the selected period (1W, 2W, 3M, etc.), trend tab index, and volatility tab index.
    PARAMETERS:
        - period_value (str): The selected period for the trend and volatility analysis (e.g., '1W', '3M', '5Y', 'All').
        - trend_tab_index (int): The index of the selected trend analysis tab.
        - volatility_tab_index (int): The index of the selected volatility analysis tab.
        - selected_ticker (str): The currently selected ticker from the ticker dropdown.
    RETURNS:
        - tuple: A tuple containing:
            - Updated trend plot figure.
            - Updated volatility plot figure.
    """
    # Determine the start date based on the selected period
    if period_value == '1W':
        dash_data['g2_start_date'] = dash_data['end_date'] - pd.DateOffset(weeks=1)
    elif period_value == '2W':
        dash_data['g2_start_date'] = dash_data['end_date'] - pd.DateOffset(weeks=2)
    elif period_value == '3M':
        dash_data['g2_start_date'] = dash_data['end_date'] - pd.DateOffset(months=3)
    elif period_value == '6M':
        dash_data['g2_start_date'] = dash_data['end_date'] - pd.DateOffset(months=6)
    elif period_value == '1Y':
        dash_data['g2_start_date'] = dash_data['end_date'] - pd.DateOffset(years=1)
    elif period_value == '5Y':
        dash_data['g2_start_date'] = dash_data['end_date'] - pd.DateOffset(years=5)
    elif period_value == 'All':
        dash_data['g2_start_date'] = dash_data['time_series_data']['date'].min()
    
    # Filter the data for the trend plot based on the calculated start date and end date
    filtered_data = dash_data['time_series_data'][
        (dash_data['time_series_data']['date'] >= dash_data['g2_start_date']) & 
        (dash_data['time_series_data']['date'] <= dash_data['end_date'])
    ]
    filtered_data2 = dash_data['SPY_data'][(dash_data['SPY_data']['date'] >= dash_data['start_date']) & (dash_data['SPY_data']['date'] <= dash_data['end_date'])]
    # Create the trend plot using the filtered data and selected trend tab index
    trend_fig = create_trend_plot(filtered_data, trend_tab_index)
    
    # Create the volatility plot using the filtered data and selected volatility tab index
    volatility_fig = create_volatility_plot(filtered_data, volatility_tab_index, filtered_data2)
    
    return trend_fig, volatility_fig
##########--------------------##########--------------------##########--------------------
if __name__ == '__main__':
    app.run_server(debug=True)
