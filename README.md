# 📈 NiftyPulse: Data-Driven Stock Analysis & Visualization

## Project Overview

This project aims to provide a comprehensive solution for organizing, cleaning, analyzing, and visualizing the performance of Nifty 50 stocks over the past year. It analyzes daily stock data (open, close, high, low, volume) to generate key performance insights and offers interactive dashboards using Python (Streamlit) and Power BI to empower investors, analysts, and enthusiasts with informed decision-making capabilities.

## ✨ Features

### Business Use Cases

  * Stock Performance Ranking: Identify the top 10 best-performing (green) and top 10 worst-performing (red) stocks yearly.
  * Market Overview: Provide an overall market summary, including average stock performance and the percentage of green vs. red stocks.
  * Investment Insights: Help identify stocks with consistent growth or significant declines.
  * Decision Support: Offer insights on average prices, volatility, and overall stock behavior for traders.

### Data Analysis & Visualizations

  * Key Metrics: Calculate Top 10 Green/Loss Stocks, Market Summary (total/green/red stocks, average price, average volume).
  * Volatility Analysis: Visualize stock volatility using the standard deviation of daily returns.
  * Cumulative Return Over Time: Show the cumulative performance of top 5 stocks.
  * Sector-wise Performance: Breakdown of average yearly returns by market sector.
  * Stock Price Correlation: Heatmap visualizing correlation between stock closing prices.
  * Monthly Gainers & Losers: Provide monthly breakdowns of top 5 performing and worst performing stocks.

## 🛠️ Technologies Used

  * Languages: Python
  * Data Manipulation: Pandas
  * Database: MySQL (with SQLAlchemy and `mysql-connector-python`)
  * Visualization & Dashboard: Streamlit, Matplotlib, Seaborn, Power BI
  * Version Control: Git, GitHub

## Project Structure

```
.
├── README.md                              # This file
├── data/                                  # Raw YAML data (e.g., 2024-01/date.yaml)
│   └── processed_csv/                     # Output directory for symbol-wise CSVs
├── stock_analysis_app.py                  # Main script for data processing, database loading, and Streamlit app
├── get_unique_tickers.py                  # Utility script to extract unique stock tickers
├── sector_data.csv                        # CSV file mapping stock symbols to sectors
└── requirements.txt                       # Lists all Python libraries and their versions
```

## ⚙️ Setup and Installation

### Prerequisites

  * Python 3.8+ installed (ensure it's added to PATH during installation).
  * MySQL Server installed and running.
  * A database user with permissions to create databases/tables and insert data.
  * Power BI Desktop installed (for Power BI dashboard setup).

### Steps
1.  **Data Preparation:**

      * YAML Data: Place your raw YAML data (e.g., unzipped from `data.rar`) into a folder named `data` in the root of your project. The structure should be `data/<YEAR-MONTH>/<DATE_TIME>.yaml`.
      * `sector_data.csv`: Create a `sector_data.csv` file in the root of your project directory. It should contain at least two columns: `Symbol` (stock ticker) and `Sector` (the industry sector for that stock).
        ```csv
        Symbol,Sector
        RELIANCE,Energy
        TCS,Information Technology
        # ... include all Nifty 50 symbols and their sectors
        ```

2.  **Database Setup:**

      * Using your MySQL/PostgreSQL client (e.g., MySQL Workbench, `psql` command line), create an empty database. For example:
        ```sql
        CREATE DATABASE nifty_stocks_db;
        ```
      * Configure Database Connection: Open `stock_analysis_app.py` and modify the `DATABASE_URI` variable with your actual database credentials:
        ```python
        # For MySQL: 'mysql+mysqlconnector://your_user:your_password@localhost/your_database_name'
        # For PostgreSQL: 'postgresql+psycopg2://your_user:your_password@localhost/your_database_name'
        DATABASE_URI = 'mysql+mysqlconnector://root:YOUR_PASSWORD@localhost/nifty_stocks_db' # <--- IMPORTANT: CHANGE THIS!
        ```

3.  **Install Python Dependencies:**

      * Create a Python virtual environment (recommended):
        ```bash
        python -m venv venv
        # Activate environment:
        # On Windows: .\venv\Scripts\activate
        # On macOS/Linux: source venv/bin/activate
        ```
      * Install all required libraries. You can either use `pip install -r requirements.txt` (if you've generated one) or install them manually:
        ```bash
        pip install pandas PyYAML sqlalchemy mysql-connector-python matplotlib seaborn streamlit numpy
        # If using PostgreSQL, replace 'mysql-connector-python' with 'psycopg2-binary'
        # pip install pandas PyYAML sqlalchemy psycopg2-binary matplotlib seaborn streamlit numpy
        ```

## 🚀 How to Run the Project (Python & Streamlit)

The `stock_analysis_app.py` script executes in three distinct phases: Data Extraction, Data Cleaning & Database Loading, and running the Streamlit Dashboard. **You must run these phases sequentially, uncommenting only one section at a time** in the `if __name__ == "__main__":` block at the bottom of `stock_analysis_app.py`.

### Phase A: Data Extraction and Transformation (YAML to CSVs)

1.  Open `stock_analysis_app.py`.
2.  In the `if __name__ == "__main__":` block, **uncomment** the line for `extract_and_transform_yaml_to_csv`:
    ```python
    print("\n--- STEP A: Starting Data Extraction and Transformation ---")
    extract_and_transform_yaml_to_csv(DATA_DIR, OUTPUT_CSV_DIR)
    print("--- STEP A: Data Extraction and Transformation Finished ---\n")
    # clean_and_load_to_db(OUTPUT_CSV_DIR, DATABASE_URI) # Comment out this line
    # run_streamlit_app() # Comment out this line
    ```
3.  Save the file.
4.  Run from your terminal (ensure your virtual environment is active):
    ```bash
    python stock_analysis_app.py
    ```
    This will parse your YAML files and save symbol-wise CSVs into the `data/processed_csv` directory.

### Phase B: Data Cleaning & Loading into Database

1.  Open `stock_analysis_app.py`.
2.  In the `if __name__ == "__main__":` block, **uncomment** the line for `clean_and_load_to_db` and **comment out** the other two execution lines:
    ```python
    # extract_and_transform_yaml_to_csv(DATA_DIR, OUTPUT_CSV_DIR) # Comment out this line
    print("\n--- STEP B: Clean Data & Load into Database ---")
    clean_and_load_to_db(OUTPUT_CSV_DIR, DATABASE_URI)
    print("--- STEP B: Data Cleaning and Database Loading Finished ---\n")
    # run_streamlit_app() # Comment out this line
    ```
3.  Save the file.
4.  Run from your terminal:
    ```bash
    python stock_analysis_app.py
    ```
    This will read the generated CSVs, clean the data, and load it into your configured MySQL database.

### Phase C: Run the Streamlit Application

1.  Open `stock_analysis_app.py`.
2.  In the `if __name__ == "__main__":` block, **uncomment** the line for `run_streamlit_app` and **comment out** the other two execution lines:
    ```python
    # extract_and_transform_yaml_to_csv(DATA_DIR, OUTPUT_CSV_DIR) # Comment out this line
    # clean_and_load_to_db(OUTPUT_CSV_DIR, DATABASE_URI) # Comment out this line
    print("\n--- STEP C: Starting Streamlit Application ---")
    run_streamlit_app()
    print("--- STEP C: Streamlit Application Started ---")
    ```
3.  Save the file.
4.  Run the Streamlit application from your terminal:
    ```bash
    streamlit run stock_analysis_app.py
    ```
    Your web browser should automatically open to the interactive dashboard. Inside the Streamlit app, remember to click the **"Load Data (from Database)"** button in the sidebar to fetch the data and populate the dashboard.

-----

### 📚 Utility Script: `get_unique_tickers.py`

This helper script is useful for identifying all unique stock tickers present in your raw YAML data, which can help ensure your `sector_data.csv` is complete.

To run it:

```bash
python get_unique_tickers.py
```

-----

## 📊 Power BI Dashboard Setup

You can create dynamic and interactive dashboards in Power BI by connecting directly to your MySQL database.

### Steps to Connect Power BI

1.  **Open Power BI Desktop.**
2.  From the "Home" tab, click **"Get data"**.
3.  Choose **"More..."** to open the Get Data dialog.
4.  Search for and select **"MySQL database"** and click **"Connect"**.
5.  Enter your **Server (e.g., `localhost` or IP address)** and **Database name (e.g., `nifty_stocks_db`)**. For MySQL, choose "Import" for Data Connectivity mode.
6.  **Authentication:**
      * Select **"Database"** as the Authentication type.
      * Enter your MySQL **Username** and **Password**.
      * Click **"Connect"**.
7.  **Navigator:** Once connected, you will see a list of tables from your database. Select the `nifty_stock_data` table (and any other relevant tables like your sector data if loaded separately).
8.  Click **"Load"**. Power BI will import the data into its model.
9.  You can now start creating visualizations, setting up relationships between tables (if you load `sector_data` into Power BI as well), and building your Power BI dashboard based on the cleaned data from your SQL database.

## 🎥 Demo Video and PowerBI output

  * LinkedIn Demo Video Link: https://www.linkedin.com/posts/monica-umamageswaran_dataanalytics-python-streamlit-activity-7356656662822182913-fqIe?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE_7PqYBCyvYmCOnir7XtTdIJhnL6JtNqSA
  * PowerBI output: 2 images had been added to this project repo in the main branch itself.
