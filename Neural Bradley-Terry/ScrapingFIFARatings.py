import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

# ================= Configuration ================= #
# Ligue 1 ID on SoFIFA is 16. (Premier League=13, La Liga=53, etc.)
LEAGUE_ID = 16 

# We want seasons 2010-2011 up to 2023-2024
START_YEAR = 2010
END_YEAR = 2023 

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
# ================================================= #

def get_roster_id(season_start_year):
    """
    Calculates the SoFIFA Roster ID based on the season.
    Example: Season 2010-2011 uses FIFA 11 data -> ID 110001
    Example: Season 2022-2023 uses FIFA 23 data -> ID 230001
    """
    fifa_version = season_start_year - 2000 + 1
    # Format is usually YY0001 for the initial database (August/September update)
    return f"{fifa_version}0001"

def scrape_sofifa_season(year):
    roster_id = get_roster_id(year)
    url = f"https://sofifa.com/teams?lg={LEAGUE_ID}&r={roster_id}&set=true"
    
    print(f"Scraping FIFA ratings for Season {year}-{year+1} (Roster {roster_id})...")
    
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Failed to retrieve {url} (Status: {response.status_code})")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # SoFIFA uses a standard table structure
        table = soup.find('table')
        if not table:
            print(f"No table found for {year}")
            return []
            
        tbody = table.find('tbody')
        rows = tbody.find_all('tr')
        
        season_data = []
        
        for row in rows:
            # Find team name (inside .col-name-wide or just the link text)
            name_col = row.find('td', class_='col-name-wide')
            if not name_col:
                name_col = row.find('td', class_='col-name') # Fallback
            
            if name_col:
                team_name = name_col.find('a').text.strip()
            else:
                continue

            # Extract Ratings. These usually have specific classes:
            # col-oa (Overall), col-at (Attack), col-md (Midfield), col-df (Defense)
            try:
                # Cleaning: sometimes they have extra spans, just get text
                ova = row.find('td', class_='col-oa').text.strip()
                att = row.find('td', class_='col-at').text.strip()
                mid = row.find('td', class_='col-md').text.strip()
                defense = row.find('td', class_='col-df').text.strip()
            except AttributeError:
                # If a column is missing (rare), skip or set to NaN
                continue

            season_data.append({
                'Season_Start_Year': year,
                'Team': team_name,
                'FIFA_Overall': int(ova),
                'FIFA_Attack': int(att),
                'FIFA_Midfield': int(mid),
                'FIFA_Defense': int(defense)
            })
            
        return season_data

    except Exception as e:
        print(f"Error for {year}: {e}")
        return []

# ================= Main Execution ================= #
all_ratings = []

for year in range(START_YEAR, END_YEAR + 1):
    data = scrape_sofifa_season(year)
    all_ratings.extend(data)
    
    # Polite delay
    time.sleep(random.uniform(1, 3))

# Save
df_fifa = pd.DataFrame(all_ratings)
filename = "ligue1_fifa_ratings_2010_2023.csv"
df_fifa.to_csv(filename, index=False)

print(f"Success! Scraped {len(df_fifa)} team ratings.")
print(df_fifa.head())