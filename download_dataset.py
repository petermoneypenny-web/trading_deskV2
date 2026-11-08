import os
import argparse
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

DEFAULT_TICKERS = ["SPY", "QQQ", "AAPL", "NVDA"]
OUTPUT_FILE = "/app/data/historical_1m_30d.csv"

def download_month_dataset(tickers, days=30, output_path=OUTPUT_FILE):
    print(f"Starting chunked download for {tickers} over last {days} days...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    all_chunks = []
    current_start = start_date
    chunk_size = 6  # 6-day intervals stay safely within Yahoo's 8-day 1m cap

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_size), end_date)
        s_str = current_start.strftime("%Y-%m-%d")
        e_str = (current_end + timedelta(days=1)).strftime("%Y-%m-%d")
        
        print(f"Fetching chunk: {s_str} to {e_str}...")
        try:
            chunk_df = yf.download(
                tickers=" ".join(tickers),
                start=s_str,
                end=e_str,
                interval="1m",
                prepost=False,
                progress=False,
                threads=False,
                group_by="column"
            )
            if chunk_df is not None and not chunk_df.empty:
                all_chunks.append(chunk_df)
        except Exception as e:
            print(f"Error downloading chunk {s_str} -> {e_str}: {e}")
            
        current_start = current_end + timedelta(days=1)

    if not all_chunks:
        print("No data retrieved.")
        return

    # Combine chunks and drop duplicate timestamps
    full_df = pd.concat(all_chunks)
    full_df = full_df[~full_df.index.duplicated(keep="first")]
    full_df.sort_index(inplace=True)

    # Flatten multi-index columns for easy CSV export/import
    flat_data = {"timestamp": full_df.index.astype(str)}
    for ticker in tickers:
        for metric in ["Open", "High", "Low", "Close", "Volume"]:
            if (metric, ticker) in full_df.columns:
                flat_data[f"{ticker}_{metric.lower()}"] = full_df[(metric, ticker)].values

    export_df = pd.DataFrame(flat_data)
    export_df.to_csv(output_path, index=False)
    print(f"Saved {len(export_df)} 1-minute bars across {len(tickers)} tickers to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    
    download_month_dataset(args.tickers, args.days)
