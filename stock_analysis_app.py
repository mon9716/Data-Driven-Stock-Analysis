# Import Libraries
import pandas as pd
import yaml
import os
import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

# Path to your extracted YAML data folder (e.g., 'data' if it's in the same directory as this script)
DATA_DIR = './data'

# Directory to save processed CSVs (will be created if it doesn't exist)
OUTPUT_CSV_DIR = 'data/processed_csv'

# Your Database Connection String
# For MySQL: 'mysql+mysqlconnector://your_user:your_password@localhost/your_database_name'
DATABASE_URI = 'mysql+mysqlconnector://root:root@localhost/nifty_stocks_db'

# Path to your sector_data.csv file
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
SECTOR_DATA_PATH = os.path.join(script_dir, 'sector_data.csv')

# --- 1. Data Extraction and Transformation ---

def extract_and_transform_yaml_to_csv(data_dir, output_csv_dir):
    """
    Extracts stock data from YAML files, transforms it, and saves it to symbol-wise CSVs.
    Adjusted for YAML files containing a direct list of stock dictionaries.
    """
    os.makedirs(output_csv_dir, exist_ok=True)
    all_raw_stock_data = [] # Changed name to reflect direct stock data

    print(f"DEBUG: Looking for data in: {os.path.abspath(data_dir)}")
    for month_folder in sorted(os.listdir(data_dir)):
        month_path = os.path.join(data_dir, month_folder)
        if os.path.isdir(month_path):
            print(f"DEBUG: Entering month folder: {month_folder}")
            for date_file_name in sorted(os.listdir(month_path)):
                if date_file_name.endswith('.yaml'):
                    file_path = os.path.join(month_path, date_file_name)
                    with open(file_path, 'r') as f:
                        data = yaml.safe_load(f)
                        if isinstance(data, list) and data: # Expecting a list of dictionaries
                            all_raw_stock_data.extend(data) # Use extend to add all dicts from the list
                            print(f"DEBUG: Successfully loaded {len(data)} records from {date_file_name}")
                        elif data is None:
                            print(f"DEBUG: Loaded file {date_file_name} is empty.")
                        else:
                            print(f"DEBUG: Skipping file {date_file_name} due to unexpected root type: {type(data)} (expected list)")
        else:
            print(f"DEBUG: Skipping non-directory item: {month_folder}")

    print(f"DEBUG: Total raw stock records loaded from all YAML files: {len(all_raw_stock_data)}")

    processed_records = []
    # Loop directly through each stock dictionary
    for stock_record in all_raw_stock_data:
        if isinstance(stock_record, dict) and stock_record:
            # Extract fields directly from the stock_record dictionary
            processed_records.append({
                'Date': stock_record.get('date'),
                'Symbol': stock_record.get('Ticker'),
                'Open': stock_record.get('open'),
                'High': stock_record.get('high'),
                'Low': stock_record.get('low'),
                'Close': stock_record.get('close'),
                'Volume': stock_record.get('volume')
            })
        else:
            print(f"DEBUG: Skipping invalid stock_record (not a dict or empty): {stock_record}")

    print(f"DEBUG: Number of records prepared for DataFrame after processing: {len(processed_records)}")
    df_nifty = pd.DataFrame(processed_records)

    if not df_nifty.empty:
        # Convert 'Date' column to datetime objects
        df_nifty['Date'] = pd.to_datetime(df_nifty['Date'])

        # Group by 'Symbol' and save each to a separate CSV
        for symbol in df_nifty['Symbol'].unique():
            symbol_df = df_nifty[df_nifty['Symbol'] == symbol].copy()
            symbol_df.sort_values(by='Date', inplace=True)
            output_path = os.path.join(output_csv_dir, f'{symbol}.csv')
            symbol_df.to_csv(output_path, index=False)
            print(f"Saved {symbol} data to {output_path}")
        print("Data extraction and transformation complete.")
    else:
        print("WARNING: No data was extracted from YAML files. No CSVs will be created.")

    # Convert 'Date' column to datetime objects
    # This line will only run if df_nifty is not empty.
    if not df_nifty.empty: # Added a check here for safety
        df_nifty['Date'] = pd.to_datetime(df_nifty['Date'])

        # Save data for each symbol to a separate CSV
        for symbol in df_nifty['Symbol'].unique():
            symbol_df = df_nifty[df_nifty['Symbol'] == symbol].copy()
            symbol_df.sort_values(by='Date', inplace=True)
            output_path = os.path.join(output_csv_dir, f'{symbol}.csv')
            symbol_df.to_csv(output_path, index=False)
            print(f"Saved {symbol} data to {output_path}")
        print("Data extraction and transformation complete.")
    else:
        print("WARNING: No data was extracted from YAML files. No CSVs will be created.") # DEBUG


# --- 2. Data Cleaning and Loading into Database ---

def clean_and_load_to_db(csv_dir, db_uri):
    """
    Reads processed CSVs, performs cleaning, and loads data into a SQL database.
    """
    engine = create_engine(db_uri)

    for filename in os.listdir(csv_dir):
        if filename.endswith('.csv'):
            symbol = os.path.splitext(filename)[0]
            file_path = os.path.join(csv_dir, filename)
            df = pd.read_csv(file_path)

            # Data Cleaning Steps
            df['Date'] = pd.to_datetime(df['Date'])

            # Handle missing values
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)

            # Remove duplicates (if any) based on Date and Symbol
            df.drop_duplicates(subset=['Date', 'Symbol'], inplace=True)

            # Ensure numeric types for financial columns
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=numeric_cols, inplace=True)

            # Load to SQL Database
            table_name = 'nifty_stock_data'
            try:
                df.to_sql(table_name, con=engine, if_exists='append', index=False)
                print(f"Loaded {symbol} data to database table '{table_name}'.")
            except sqlalchemy.exc.IntegrityError as e:
                print(f"Skipping duplicate data for {symbol}: {e}")
            except Exception as e:
                print(f"Error loading {symbol} to DB: {e}")

    print("Data cleaning and loading to database complete.")


# --- 3. Data Analysis Functions ---

def load_data_from_db(db_uri, table_name='nifty_stock_data'):
    """Loads all stock data from the database."""
    engine = create_engine(db_uri)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, con=engine, parse_dates=['Date'])
    return df

def calculate_yearly_returns(df):
    """Calculates yearly return for each stock."""
    yearly_returns = {}
    for symbol in df['Symbol'].unique():
        symbol_df = df[df['Symbol'] == symbol].sort_values(by='Date')
        if not symbol_df.empty:
            start_price = symbol_df.iloc[0]['Close']
            end_price = symbol_df.iloc[-1]['Close']
            if start_price > 0:
                yearly_returns[symbol] = ((end_price - start_price) / start_price) * 100
            else:
                yearly_returns[symbol] = 0
    return pd.Series(yearly_returns, name='Yearly_Return')

def get_top_n_stocks(df_returns, n=10, ascending=False):
    """Returns top N performing or loss stocks."""
    return df_returns.sort_values(ascending=ascending).head(n)

def get_market_summary(df, df_returns):
    """Provides overall market summary."""
    # Ensure these are standard Python ints, not NumPy ints, for cleaner JSON/display
    green_stocks = int((df_returns > 0).sum())
    red_stocks = int((df_returns <= 0).sum())
    total_stocks = int(len(df_returns))

    # Ensure these are standard Python floats, not NumPy floats
    avg_price_all_stocks = float(df['Close'].mean())
    avg_volume_all_stocks = float(df['Volume'].mean())

    return {
        'Total Stocks': total_stocks,
        'Green Stocks': green_stocks,
        'Red Stocks': red_stocks,
        'Percentage Green': round((green_stocks / total_stocks) * 100, 2) if total_stocks > 0 else 0.0,
        'Percentage Red': round((red_stocks / total_stocks) * 100, 2) if total_stocks > 0 else 0.0,
        'Average Close Price (All Stocks)': avg_price_all_stocks,
        'Average Volume (All Stocks)': avg_volume_all_stocks
    }

def calculate_volatility(df):
    """Calculates daily returns and volatility (standard deviation of daily returns)."""
    df_copy = df.copy()
    df_copy['Daily_Return'] = df_copy.groupby('Symbol')['Close'].pct_change()
    volatility = df_copy.groupby('Symbol')['Daily_Return'].std().dropna() * np.sqrt(252)
    return volatility.sort_values(ascending=False)

def calculate_cumulative_return(df):
    """Calculates cumulative return for each stock."""
    df_sorted = df.sort_values(by=['Symbol', 'Date'])
    # Handle cases where the first Close price might be 0 or NaN, causing division by zero
    df_sorted['Daily_Return'] = df_sorted.groupby('Symbol')['Close'].pct_change().fillna(0)
    # Calculate cumulative product, then subtract 1 to get cumulative return
    df_sorted['Cumulative_Return'] = (1 + df_sorted['Daily_Return']).cumprod() - 1
    return df_sorted

def get_sector_performance(df, sector_data_path):
    """Calculates average yearly return by sector."""
    # --- ADD THIS LINE FOR DEBUGGING ---
    #st.info(f"DEBUG: Attempting to load sector data from: `{sector_data_path}`")
    
    try:
        df_sector = pd.read_csv(sector_data_path)
        # Ensure Symbol column in sector data is consistent (e.g., uppercase)
        df_sector['Symbol'] = df_sector['Symbol'].str.upper()
    except FileNotFoundError:
        st.error(f"Error: Sector data file not found at `{sector_data_path}`. Please check the path.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading sector data: {e}")
        return pd.DataFrame()

    # Calculate yearly returns for each stock first
    # Ensure start_price is not zero to avoid division by zero
    df_with_returns = df.groupby('Symbol').apply(lambda x: (x.iloc[-1]['Close'] - x.iloc[0]['Close']) / x.iloc[0]['Close'] * 100 if x.iloc[0]['Close'] > 0 else 0)
    df_with_returns = df_with_returns.rename('Yearly_Return').reset_index()

    # Merge stock returns with sector data
    df_merged = pd.merge(df_with_returns, df_sector, on='Symbol', how='left')

    # Handle stocks not found in sector_data.csv
    if df_merged['Sector'].isnull().any():
        missing_symbols = df_merged[df_merged['Sector'].isnull()]['Symbol'].tolist()
        st.warning(f"The following symbols were not found in '{os.path.basename(sector_data_path)}': {', '.join(missing_symbols)}. They will be excluded from sector analysis.")
        df_merged.dropna(subset=['Sector'], inplace=True) # Remove rows with missing sector

    if not df_merged.empty:
        sector_avg_returns = df_merged.groupby('Sector')['Yearly_Return'].mean().sort_values(ascending=False)
        return sector_avg_returns
    else:
        return pd.DataFrame()


def calculate_correlation_matrix(df):
    """Calculates and returns the correlation matrix of stock closing prices."""
    # Ensure there's enough data for pivoting and correlation
    if df.empty or df['Date'].nunique() < 2 or df['Symbol'].nunique() < 2:
        return pd.DataFrame() # Return empty if not enough data
    df_pivot = df.pivot(index='Date', columns='Symbol', values='Close')
    correlation_matrix = df_pivot.corr()
    return correlation_matrix

def get_monthly_gainers_losers(df):
    """Identifies top 5 gainers and losers for each month."""
    df_copy = df.copy()
    df_copy['YearMonth'] = df_copy['Date'].dt.to_period('M')
    monthly_performance = {}

    # Ensure data is sorted by date within each symbol group before calculating returns
    df_copy = df_copy.sort_values(by=['Symbol', 'Date'])

    for ym in sorted(df_copy['YearMonth'].unique()):
        monthly_df = df_copy[df_copy['YearMonth'] == ym].copy()
        if not monthly_df.empty:
            # Calculate monthly return for each stock: (End_of_month_close - Start_of_month_close) / Start_of_month_close
            # Get first and last close price for each stock in the month
            first_close = monthly_df.groupby('Symbol')['Close'].first()
            last_close = monthly_df.groupby('Symbol')['Close'].last()

            monthly_return = ((last_close - first_close) / first_close) * 100
            monthly_return = monthly_return.replace([np.inf, -np.inf], np.nan).dropna() # Handle division by zero

            if not monthly_return.empty:
                top_gainers = monthly_return.sort_values(ascending=False).head(5)
                top_losers = monthly_return.sort_values(ascending=True).head(5)

                monthly_performance[str(ym)] = {
                    'Gainers': top_gainers,
                    'Losers': top_losers
                }
    return monthly_performance


# --- 4. Visualization Functions (Matplotlib/Seaborn for local, Streamlit for app) ---

def plot_bar_chart(data, title, xlabel, ylabel, color='skyblue', figsize=(12, 6)):
    """Generic bar chart plotting function."""
    fig, ax = plt.subplots(figsize=figsize)
    if isinstance(data, pd.Series):
        data.plot(kind='bar', ax=ax, color=color)
    elif isinstance(data, pd.DataFrame):
        data.plot(kind='bar', ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_line_chart(df_plot, title, xlabel, ylabel, figsize=(14, 7)):
    """Generic line chart plotting function for cumulative returns."""
    fig, ax = plt.subplots(figsize=figsize)
    for col in df_plot.columns:
        ax.plot(df_plot.index, df_plot[col], label=col)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_heatmap(data, title, figsize=(10, 8), annot=False, fmt=".2f"): 
    """Heatmap plotting function."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data, annot=annot, cmap='coolwarm', fmt=fmt, linewidths=.5, ax=ax, annot_kws={"size": 8}) 
    ax.set_title(title)
    plt.xticks(rotation=90, ha='right') 
    plt.yticks(rotation=0)              
    plt.tight_layout()
    return fig

# --- 5. Streamlit Application ---

def run_streamlit_app():
    st.set_page_config(layout="wide", page_title="NiftyPulse: Advanced Nifty 50 Analysis")
    st.title("🚀 NiftyPulse: Advanced Nifty 50 Analysis")
    st.write("Dive deep into Nifty 50 stock performance, discover trends, and unlock valuable investment insights.")

    # --- Sidebar for Data Management and Customization ---
    st.sidebar.markdown("## ⚙️ App Controls")

    # Initialize df_stocks in session state to persist it
    if 'df_stocks' not in st.session_state:
        st.session_state.df_stocks = pd.DataFrame()

    st.sidebar.subheader("🗄️ Data Management")
    if st.sidebar.button("Load Data (from Database)", use_container_width=True):
        with st.spinner('Loading historical stock data...'):
            try:
                st.session_state.df_stocks = load_data_from_db(DATABASE_URI)
                st.sidebar.success("✅ Data loaded successfully!")
                st.sidebar.info(f"Total records loaded: **{len(st.session_state.df_stocks)}**.")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading data: {e}. Please ensure your database is running and `DATABASE_URI` is correct.")

    # Check if data is loaded before proceeding with analysis
    if st.session_state.df_stocks.empty:
        st.warning("Please click 'Load Data (from Database)' in the sidebar to begin analysis.")
        st.stop() # Stop execution until data is loaded

    df_stocks = st.session_state.df_stocks.copy() 

    # Pre-calculate common metrics for display
    df_yearly_returns = calculate_yearly_returns(df_stocks)

    # MOVED market_summary DEFINITION HERE, BEFORE ITS USAGE
    market_summary = get_market_summary(df_stocks, df_yearly_returns) 

    df_volatility = calculate_volatility(df_stocks)
    df_cumulative = calculate_cumulative_return(df_stocks)
    df_correlation = calculate_correlation_matrix(df_stocks)
    monthly_perf = get_monthly_gainers_losers(df_stocks)


    # --- Display Sections ---
    st.header("📊 Overall Market Summary")
    
    col_total, col_green, col_red, col_avg_price, col_avg_volume = st.columns(5)
    
    with col_total:
        st.metric(label="Total Stocks", value=market_summary['Total Stocks'])
    with col_green:
        st.metric(label="Green Stocks", value=f"{market_summary['Green Stocks']} ({market_summary['Percentage Green']:.1f}%)", delta="Positive Return")
    with col_red:
        st.metric(label="Red Stocks", value=f"{market_summary['Red Stocks']} ({market_summary['Percentage Red']:.1f}%)", delta="Negative Return", delta_color="inverse")
    with col_avg_price:
        st.metric(label="Avg. Close Price", value=f"₹{market_summary['Average Close Price (All Stocks)']:.2f}")
    with col_avg_volume:
        st.metric(label="Avg. Volume", value=f"{market_summary['Average Volume (All Stocks)']:.0f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Top 10 Best Performing Stocks (Yearly Return)")
        top_gainers_yearly = get_top_n_stocks(df_yearly_returns, 10, ascending=False)
        st.dataframe(top_gainers_yearly.reset_index().rename(columns={'index': 'Symbol'}), use_container_width=True)
        fig_gainers_yearly = plot_bar_chart(top_gainers_yearly, 'Top 10 Best Performing Stocks', 'Symbol', 'Yearly Return (%)', color='lightgreen', figsize=(10, 6))
        st.pyplot(fig_gainers_yearly)

    with col2:
        st.subheader("📉 Top 10 Worst Performing Stocks (Yearly Return)")
        top_losers_yearly = get_top_n_stocks(df_yearly_returns, 10, ascending=True)
        st.dataframe(top_losers_yearly.reset_index().rename(columns={'index': 'Symbol'}), use_container_width=True)
        fig_losers_yearly = plot_bar_chart(top_losers_yearly, 'Top 10 Worst Performing Stocks', 'Symbol', 'Yearly Return (%)', color='salmon', figsize=(10, 6))
        st.pyplot(fig_losers_yearly)


    st.markdown("---")
    st.header("⚡ Volatility Analysis")
    st.subheader("Top 10 Most Volatile Stocks")
    top_10_volatile = df_volatility.head(10)
    st.dataframe(top_10_volatile.reset_index().rename(columns={'index': 'Symbol', 0: 'Volatility'}), use_container_width=True)
    fig_volatility = plot_bar_chart(top_10_volatile, 'Top 10 Most Volatile Stocks', 'Symbol', 'Volatility (Annualized Std Dev of Daily Returns)', color='orange', figsize=(10, 6))
    st.pyplot(fig_volatility)


    st.markdown("---")
    st.header("🚀 Cumulative Return Over Time")
    st.subheader("Top 5 Performing Stocks (Cumulative Return)")
    latest_cumulative_returns = df_cumulative.groupby('Symbol')['Cumulative_Return'].last().sort_values(ascending=False)
    top_5_symbols_cumulative = latest_cumulative_returns.head(5).index.tolist()

    if top_5_symbols_cumulative:
        df_plot_cumulative = df_cumulative[df_cumulative['Symbol'].isin(top_5_symbols_cumulative)].pivot(index='Date', columns='Symbol', values='Cumulative_Return')
        fig_cumulative = plot_line_chart(df_plot_cumulative, 'Cumulative Return for Top 5 Performing Stocks', 'Date', 'Cumulative Return')
        st.pyplot(fig_cumulative)
    else:
        st.info("No data to plot cumulative returns for top 5 stocks.")


    st.markdown("---")
    st.header("🏢 Sector-wise Performance")
    sector_perf = get_sector_performance(df_stocks, SECTOR_DATA_PATH)
    if not sector_perf.empty:
        st.subheader("Average Yearly Return by Sector")
        st.dataframe(sector_perf.reset_index().rename(columns={'index': 'Sector', 0: 'Average Yearly Return'}), use_container_width=True)
        fig_sector = plot_bar_chart(sector_perf, 'Average Yearly Return by Sector', 'Sector', 'Average Yearly Return (%)', color='lightgreen')
        st.pyplot(fig_sector)
    else:
        st.warning("Could not perform Sector-wise Performance analysis. Please check sector_data.csv and the path.")


    st.markdown("---")
    st.header("🔗 Stock Price Correlation")
    st.subheader("Correlation Heatmap of Stock Closing Prices")
    
    if not df_correlation.empty:
        all_symbols_for_corr = df_correlation.columns.tolist()

        default_selection = all_symbols_for_corr[:10] if len(all_symbols_for_corr) > 10 else all_symbols_for_corr

        selected_symbols_corr = st.multiselect(
            "Select symbols for correlation heatmap (select up to 15 for optimal view):",
            all_symbols_for_corr,
            default=default_selection,
            help="Choose which stock symbols to include in the correlation heatmap. Fewer stocks yield a more readable plot."
        )

        if selected_symbols_corr:
            if len(selected_symbols_corr) > 15:
                st.warning("Too many symbols selected. Displaying correlation for the first 15 selected symbols for readability.")
                symbols_to_plot = selected_symbols_corr[:15]
            else:
                symbols_to_plot = selected_symbols_corr
            
            filtered_correlation_matrix = df_correlation.loc[symbols_to_plot, symbols_to_plot]

            should_annotate = len(symbols_to_plot) <= 5
            
            fig_correlation = plot_heatmap(
                filtered_correlation_matrix,
                'Stock Price Correlation Heatmap',
                figsize=(max(8, len(symbols_to_plot)), max(7, len(symbols_to_plot) * 0.8)),
                annot=should_annotate,
                fmt=".2f"
            )
            st.pyplot(fig_correlation)
        else:
            st.info("Select symbols from the dropdown above to view the correlation heatmap.")
    else:
        st.info("Not enough data to compute stock price correlation.")


    st.markdown("---")
    st.header("🗓️ Monthly Gainers and Losers")
    months = list(monthly_perf.keys())
    if months:
        selected_month = st.selectbox("Select a Month", months, help="Choose a month to see top 5 gainers and losers.")
        if selected_month:
            st.subheader(f"Top 5 Gainers and Losers for {selected_month}")
            col_g, col_l = st.columns(2)
            with col_g:
                st.write("📈 **Top 5 Gainers:**")
                if not monthly_perf[selected_month]['Gainers'].empty:
                    st.dataframe(monthly_perf[selected_month]['Gainers'].reset_index().rename(columns={'index': 'Symbol', 0: 'Monthly Return (%)'}), use_container_width=True)
                else:
                    st.info("No gainers for this month.")
            with col_l:
                st.write("📉 **Top 5 Losers:**")
                if not monthly_perf[selected_month]['Losers'].empty:
                    st.dataframe(monthly_perf[selected_month]['Losers'].reset_index().rename(columns={'index': 'Symbol', 0: 'Monthly Return (%)'}), use_container_width=True)
                else:
                    st.info("No losers for this month.")
    else:
        st.info("No monthly performance data available yet. Ensure data covers multiple months.")


    st.markdown("---")
    st.caption("Dashboard powered by Streamlit | Data-Driven Stock Analysis | Developed by Your Name/Team Name")

# --- Main Execution Block (RUN THESE STEPS ONE BY ONE!) ---
if __name__ == "__main__":
    # print("\n--- STEP A: Starting Data Extraction and Transformation ---")
    #extract_and_transform_yaml_to_csv(DATA_DIR, OUTPUT_CSV_DIR)
    # print("--- STEP A: Data Extraction and Transformation Finished ---\n")

    # >>> Step B: Clean Data & Load into Database <<<
    #clean_and_load_to_db(OUTPUT_CSV_DIR, DATABASE_URI)
    # print("--- STEP B: Data Cleaning and Database Loading Finished ---\n")

    # >>> Step C: Run the Streamlit Application <<<
    # print("\n--- STEP C: Starting Streamlit Application ---")
    run_streamlit_app()
    # print("--- STEP C: Streamlit Application Started ---")