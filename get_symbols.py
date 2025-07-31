import os
import yaml

def get_unique_tickers(data_path='data'):
    unique_tickers = set() # Use a set to automatically handle uniqueness
    print(f"Checking data_path: {data_path}")
    
    try:
        data_contents = os.listdir(data_path)
        print(f"Contents of '{data_path}': {data_contents}")
    except FileNotFoundError:
        print(f"Error: The 'data' folder was not found at {os.path.abspath(data_path)}")
        return []
    except Exception as e:
        print(f"An error occurred listing contents of '{data_path}': {e}")
        return []

    for year_folder in sorted(data_contents): # Iterate through the contents found
        year_path = os.path.join(data_path, year_folder)
        
        # Check if it's a directory (and not data.rar or other files)
        if os.path.isdir(year_path):
            print(f"Entering year folder: {year_path}")
            for daily_file in sorted(os.listdir(year_path)):
                if daily_file.endswith('.yaml'):
                    file_path = os.path.join(year_path, daily_file)
                    print(f"Attempting to open file: {file_path}")
                    try:
                        with open(file_path, 'r') as f:
                            daily_stock_data = yaml.safe_load(f)
                            # Check if daily_stock_data is a list of dictionaries
                            if isinstance(daily_stock_data, list):
                                for stock_entry in daily_stock_data:
                                    # --- CRITICAL CHANGE HERE: 'Ticker' (capital T) ---
                                    if isinstance(stock_entry, dict) and 'Ticker' in stock_entry:
                                        unique_tickers.add(stock_entry['Ticker'])
                            # Handle cases where the YAML might be a single dictionary
                            elif isinstance(daily_stock_data, dict) and 'Ticker' in daily_stock_data:
                                unique_tickers.add(daily_stock_data['Ticker'])
                    except yaml.YAMLError as exc:
                        print(f"Error parsing YAML file {file_path}: {exc}")
                    except Exception as e:
                        print(f"An unexpected error occurred with file {file_path}: {e}")
        else:
            print(f"Skipping non-directory item in 'data' folder: {year_path}")

    return sorted(list(unique_tickers)) # Convert to list and sort for readability

if __name__ == '__main__':
    print("Extracting unique tickers...")
    tickers = get_unique_tickers()
    if tickers:
        print("\nFound unique tickers:")
        for ticker in tickers:
            print(ticker)
    else:
        print("No tickers found. Check your data path.")