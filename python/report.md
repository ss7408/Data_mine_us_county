# Report: US County Household Clustering and Trend Analysis

## Summary of the Project
This project gathers household demographic data from the U.S. Census API (group B11002) across all U.S. counties from 2009 to 2023, and it merges it with geographic shapefile data and analyzes household composition trends over time. 
The primary goals are (1) to cluster counties based on normalized household statistics using unsupervised learning (KMeans), and 
(2) to extract time-series features such as slope, acceleration, and trend stability for married and unmarried households. 
The final dataset enables visual exploration of temporal and geographic household dynamics.


## Implemented Technical Fix

I modified the Python code under the `python/` folder to limit clustering to the most recent year of data (`year == 2023`). 
I then scaled the household demographic variables using `StandardScaler` and applied `KMeans` clustering on the filtered dataset. 
This change dramatically improves runtime and maintains a realistic snapshot of current household composition across counties.

### Change Summary (Commented in Code):
```python
# Limit clustering to most recent year to avoid distortion from outdated data
latest_year = bdf['year'].max()
latest_df = bdf[bdf['year'] == latest_year].copy()
...
kmeans.fit_predict(latest_df[numeric_cols])
