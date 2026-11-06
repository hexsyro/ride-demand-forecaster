
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def load(df, labels):
    df['weather']=df['weather_code'].map(labels['weather_code'])
    df=df.drop(columns=['weather_code'])
    df['season']=df['season'].map(labels['season'])

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    return df

def encode(df):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    cols = ["weather", "season"]
    cat = pd.DataFrame(
        ohe.fit_transform(df[cols]),
        columns=ohe.get_feature_names_out(cols),
        index=df.index)

    df = pd.concat([df.drop(columns=cols), cat], axis=1)
    return df

def split_target(df):
    x=df.drop(columns=['cnt'])
    y=df['cnt']
    return x, y



def main():
    print()
    print()
    labels = {
        'weather_code': {
            1: 'Clear',
            2: 'Scattered clouds',
            3: 'Broken clouds',
            4: 'Cloudy',
            7: 'Light rain',
            10: 'Rain with thunderstorm',
            26: 'Snowfall',
            94: 'Freezing fog'
        },
        'season': {
            0: 'Spring',
            1: 'Summer',
            2: 'Fall',
            3: 'Winter'
        }
    }

    print('Loading data...')
    df=pd.read_csv('data/raw/store_sharing.csv')
    df=load(df, labels)
    df.info()
    print()

    print('Encoding weather and season columns...')
    df = encode(df)
    df.info()
    print()

    print('Splitting into input and target...')
    x, y =split_target(df)
    print()

    print('Saving encoded files...')
    x.to_csv('data/processed/encoded_ride_data.csv', index=False)
    y.to_csv('data/processed/ride_data_target.csv', index=False)
    print('  Saved 2 files to data/processed/')

    print('Done!')
    

if __name__ == '__main__':
   main()