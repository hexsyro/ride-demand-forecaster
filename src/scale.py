import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale(df,num_cols):
    sc=StandardScaler()
    sc.fit(df[num_cols])

    df_scaled = df.copy()
    df_scaled[num_cols] = sc.transform(df[num_cols])
    return df_scaled


def main():
    print()
    print()
    num_cols = ['t1', 't2', 'hum', 'wind_speed']

    print('Loading data...')
    df=pd.read_csv('data/processed/encoded_ride_data.csv')
    print()

    print('Scaling numeric columns...')
    df_scaled=scale(df,num_cols)
    print()

    print('Saving both scaled & unscaled files...')
    df_scaled.to_csv('data/processed/scaled_ride_data.csv', index=False)
    df.to_csv('data/processed/unscaled_ride_data.csv', index=False)
    print('  Saved 2 files to data/processed/')

    print('Done!')


if __name__ == '__main__':
   main()