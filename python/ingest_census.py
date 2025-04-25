import csv
import os
import re

import pandas as pd
import geopandas as gpd
import requests

# Load an environment variable in ~/.bashrc
# https://api.census.gov/data/key_signup.html
key = os.environ.get('CENSUS_KEY')

URL = 'https://api.census.gov/data/{year}/acs/acs5'
payload = {
        'get': 'group(B11002)',
        'for': 'county:*',
        'key': key}

yr_range = range(2009, 2024)

dfs = []
for yr in yr_range:
    resp = requests.get(URL.format(year=yr), params=payload)
    assert resp.status_code == 200
    dat = resp.json()
    df = pd.DataFrame(dat[1:], columns=dat[0])
    df['year'] = yr
    dfs.append(df)

df = pd.concat(dfs)

# Download the latest geometries for countries files into a folder called `geos/`
# https://www.census.gov/cgi-bin/geo/shapefiles/index.php
geos = gpd.read_file('geos')
geos = geos.loc[:, ['GEOID', 'INTPTLAT', 'INTPTLON']]
geos.INTPTLAT = geos.INTPTLAT.astype(float)
geos.INTPTLON = geos.INTPTLON.astype(float)

df['GEOID'] = df.GEO_ID.apply(lambda x: re.sub(r'.+US', '', x))
bdf = df.merge(geos, left_on='GEOID', right_on='GEOID')
bdf.drop(columns=['GEOID', 'GEO_ID'], inplace=True)

bdf.to_csv('with_geo_household_cnt.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)



from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.express as px

# --- Filtering latest year only ---
latest_year = bdf['year'].max()
latest_df = bdf[bdf['year'] == latest_year].copy()

# --- Normalize the numeric household columns ---
numeric_cols = [col for col in bdf.columns if col.startswith('B11002_')]
latest_df[numeric_cols] = latest_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
latest_df = latest_df.dropna(subset=numeric_cols)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(latest_df[numeric_cols])

# --- Find optimal number of clusters with Elbow method ---
inertias = []
K_range = range(2, 11)
for k in tqdm(K_range, desc="Finding optimal k"):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# --- Run final KMeans with chosen k ---
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
latest_df['cluster'] = kmeans.fit_predict(X_scaled)

# --- Plot clusters on interactive US map ---
fig = px.scatter_geo(
    latest_df,
    lat='INTPTLAT',
    lon='INTPTLON',
    color='cluster',
    title=f'County Clusters Based on Household Demographics ({latest_year})',
    projection='albers usa',
    hover_name='cluster',
    color_continuous_scale='Turbo'
)
fig.update_layout(height=600)
fig.show()