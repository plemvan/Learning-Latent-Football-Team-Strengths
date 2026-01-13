import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

#Config
START_YEAR = 2010
END_YEAR = 2025  
BASE_URL = "https://www.transfermarkt.com/ligue-1/startseite/wettbewerb/FR1/plus/?saison_id={}"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

#Functions to scrape the market value from Transfermarkt

def parse_value(value_str):
    """
    Converts strings like '€1.20bn' or '€300.00m' into a float (in Millions).
    Returns value in Million Euros.
    """
    if not value_str or value_str == "-":
        return 0.0
    
    clean_str = value_str.replace('€', '').strip()
    
    if 'bn' in clean_str:
        return float(clean_str.replace('bn', '')) * 1000
    elif 'm' in clean_str:
        return float(clean_str.replace('m', ''))
    elif 'k' in clean_str:
        return float(clean_str.replace('k', '')) / 1000
    
    return 0.0

def scrape_season(year):
    url = BASE_URL.format(year)
    print(f"Scraping {year}-{year+1} season...")
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status() 
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('div', {'id': 'yw1'})
        
        if not table:
            print(f"Could not find table for {year}")
            return []

        teams_data = []

        rows = table.find_all('tr', class_=['odd', 'even'])
        
        for row in rows:
            cols = row.find_all('td')
            
            team_name_tag = row.find('td', class_='hauptlink no-border-links')
            if not team_name_tag:
                continue
            team_name = team_name_tag.text.strip()
            value_tag = row.find_all('td', class_='rechts')
            
            if value_tag:
                raw_value = value_tag[-1].text.strip() 
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

# Main scraping loop

all_data = []
for year in range(START_YEAR, END_YEAR + 1):
    season_data = scrape_season(year)
    all_data.extend(season_data)
    sleep_time = random.uniform(2, 5)
    print(f"Found {len(season_data)} teams. Sleeping for {sleep_time:.2f}s...")
    time.sleep(sleep_time)

df = pd.DataFrame(all_data)
filename = "ligue1_market_values_2010_2025.csv"
df.to_csv(filename, index=False)
print(f"Done! Saved {len(df)} rows to {filename}")
print(df.head())