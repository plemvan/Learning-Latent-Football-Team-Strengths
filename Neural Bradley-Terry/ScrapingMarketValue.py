import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re

# ================= Configuration ================= #
START_YEAR = 2010
END_YEAR = 2023  # The 2023-2024 season
BASE_URL = "https://www.transfermarkt.com/ligue-1/startseite/wettbewerb/FR1/plus/?saison_id={}"

# Headers are CRITICAL. Without this, Transfermarkt will block you.
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
# ================================================= #

def parse_value(value_str):
    """
    Converts strings like '€1.20bn' or '€300.00m' into a float (in Millions).
    Returns value in Million Euros.
    """
    if not value_str or value_str == "-":
        return 0.0
    
    # Remove currency symbol and whitespace
    clean_str = value_str.replace('€', '').strip()
    
    if 'bn' in clean_str:
        # Convert Billions to Millions (e.g., 1.20bn -> 1200.0)
        return float(clean_str.replace('bn', '')) * 1000
    elif 'm' in clean_str:
        # Keep Millions as is (e.g., 300.00m -> 300.0)
        return float(clean_str.replace('m', ''))
    elif 'k' in clean_str:
        # Convert Thousands to Millions (e.g., 500k -> 0.5)
        return float(clean_str.replace('k', '')) / 1000
    
    return 0.0

def scrape_season(year):
    url = BASE_URL.format(year)
    print(f"Scraping {year}-{year+1} season...")
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status() # Check for 403/404 errors
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the main table containing team values
        # The ID is usually specific to the competition structure
        table = soup.find('div', {'id': 'yw1'})
        
        if not table:
            print(f"Could not find table for {year}")
            return []

        teams_data = []
        
        # Iterate over table rows (odd and even)
        rows = table.find_all('tr', class_=['odd', 'even'])
        
        for row in rows:
            cols = row.find_all('td')
            
            # Helper to find the team name (usually in the column with 'hauptlink')
            team_name_tag = row.find('td', class_='hauptlink no-border-links')
            if not team_name_tag:
                continue
            team_name = team_name_tag.text.strip()
            
            # Helper to find market value (usually the last column or specific class)
            # On Transfermarkt list view, total value is often the last link or bold text in the last column
            value_tag = row.find_all('td', class_='rechts')
            
            if value_tag:
                # The total market value is usually the LAST 'rechts' column in the row
                raw_value = value_tag[-1].text.strip() # e.g., "€1.05bn"
                market_val = parse_value(raw_value)
            else:
                market_val = 0.0

            teams_data.append({
                'Season_Start_Year': year,
                'Season': f"{year}-{year+1}",
                'Team': team_name,
                'Total_Market_Value_Millions': market_val
            })
            
        return teams_data

    except Exception as e:
        print(f"Error scraping {year}: {e}")
        return []

# ================= Main Execution ================= #
all_data = []

for year in range(START_YEAR, END_YEAR + 1):
    season_data = scrape_season(year)
    all_data.extend(season_data)
    
    # IMPORTANT: Sleep to be polite and avoid bans
    sleep_time = random.uniform(2, 5)
    print(f"Found {len(season_data)} teams. Sleeping for {sleep_time:.2f}s...")
    time.sleep(sleep_time)

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Save to CSV
filename = "ligue1_market_values_2010_2023.csv"
df.to_csv(filename, index=False)
print(f"Done! Saved {len(df)} rows to {filename}")

# Preview
print(df.head())