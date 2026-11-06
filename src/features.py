import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


def poly_feat(df, num_cols):
    pf = PolynomialFeatures(degree=2, include_bias=False)
    pf.set_output(transform='pandas')

    poly=pf.fit_transform(df[num_cols])
    new = poly.drop(columns=num_cols)
    return df.join(new)

def timestamp_col(df):
    df=df.copy()
    c = pd.to_datetime(df['timestamp'])
    t = c.loc[df.index]
    df['day_of_week'] = t.dt.dayofweek
    df['month'] = t.dt.month

    start = c.min()
    t = c.loc[df.index]
    df['is_weekend'] = (t.dt.dayofweek >= 5).astype(int)
    df['is_rush_hour'] = t.dt.hour.isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df['days_since_start'] = (t - start).dt.days
    return df.drop(columns=['timestamp'])

def plot_rides_by_hour(df,y):
    plot_df = df.join(y)

    ax = plot_df.groupby(['hour'])['cnt'].mean().plot(figsize=(9, 5),marker='o', linewidth=2.5)
    ax.set_xticks(range(24))
    ax.set_xlabel('hour of day')
    ax.set_ylabel('average ride count')
    ax.set_title('Average rides by hour of day')
    ax.grid(alpha=0.3)

    plt.savefig('outputs/figures/hour_of_day.png', dpi=200, bbox_inches='tight')
    plt.show()

def plot_rides_by_days_since(df,y):
    plot_df = df.join(y)
    daily = plot_df.groupby(['days_since_start'])['cnt'].mean()
    full_range = range(plot_df['days_since_start'].min(), plot_df['days_since_start'].max() + 1)
    daily = daily.reindex(full_range)

    ax = daily.plot(alpha=0.4, figsize=(10, 5.6))
    daily.rolling(30, center=True).mean().plot(ax=ax, linewidth=2.5)
    ax.set_xlabel('Days Since Start')
    ax.set_ylabel('average ride count')
    ax.set_title('Average rides by Days Since Start')
    ax.grid(alpha=0.3)

    slope, intercept = np.polyfit(daily.dropna().index, daily.dropna().values, 1)
    ax.plot(daily.index, slope * daily.index + intercept, color='black', linestyle='--')

    plt.savefig('outputs/figures/days_since_start.png', dpi=200, bbox_inches='tight')
    plt.show()


def main():
    print()
    print()
    num_cols = ['t1', 't2', 'hum', 'wind_speed']

    print('Loading data...')
    df=pd.read_csv('data/processed/scaled_ride_data.csv')
    y=pd.read_csv('data/processed/ride_data_target.csv')
    print()

    print('Adding time features...')
    df = timestamp_col(df)
    
    print('Applying Apply PolynomialFeatures...')
    df = poly_feat(df, num_cols)
    print()

    print('Plotting rides by hour...')
    plot_rides_by_hour(df,y)
    print('Plotting the long-term trend...')
    plot_rides_by_days_since(df, y)    
    print('  Saved 2 plot charts to outputs/figures/')
    print()

    print('Saving feature engineeered input file...')
    df.to_csv('data/processed/fully_engineered_ride_data.csv', index=False)
    print('  Saved the file to data/processed/')

    print('Done!')
    

if __name__ == '__main__':
   main()