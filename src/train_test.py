import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


class SplitTrainTest:
    def __init__(self, df,df1,df2,df3,y):
         self.df=df
         self.df1=df1
         self.df2=df2
         self.df3=df3
         self.y=y

         self.rmse_dict={}
         self.x_data={}

    def split(self, x,y):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test

    def train(self, x_train, y_train):
            rf = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)
            rf.fit(x_train, y_train)
            return rf

    def rmse_test(self, x_test,y_test, rf, name, rmse_dict):
            RMSE = round(root_mean_squared_error(y_test, rf.predict(x_test)), 2)
            rmse_dict[name]=RMSE
            return RMSE, rmse_dict
    
    def run_pipe(self,df,df1,df2,df3,y):
        self.x_data={'Raw':df, 'Encoded':df1, 'Scaled':df2, 'Fully_engineered':df3}
        for name, data in self.x_data.items():
            print()
            print(f'[ ==== Running the Pipeline for {name} data. ==== ]')
            print('--------------------------------------------------')
            print('Saving splitted input and target file...')
            x_train, x_test, y_train, y_test=self.split(data,y)
            print('Training the model...')
            rf=self.train(x_train, y_train)
            RMSE, rmse_dict=self.rmse_test(x_test,y_test, rf, name, self.rmse_dict)
            print(f"{name} data gets,{RMSE} of a RMSE score with Random Forest Regressor.")
            joblib.dump(rf,f'models/{name}_model.joblib')
            print('Saved the model to models/ folder...')
        return rmse_dict

    def plot_rmse(self, rmse_dict, save_path='outputs/figures/rmse_by_stage.png'):
        """Draw a bar chart comparing the test RMSE of each model stage.

        Lower RMSE means the model's predictions are closer to the real
        ride counts, so the shortest bar is the best stage.
        """
        # Turn names like 'Fully_engineered' into 'Fully engineered'
        stages = [name.replace('_', ' ') for name in rmse_dict]
        scores = list(rmse_dict.values())

        # Highlight the best (lowest) score in a different color
        best = scores.index(min(scores))
        colors = ['#4C72B0'] * len(scores)
        colors[best] = '#2E8B57'

        fig, ax = plt.subplots(figsize=(9, 5.6))
        bars = ax.bar(stages, scores, color=colors, width=0.6)
        ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=11)

        ax.set_title('Bike Ride Demand: Model Error by Data Stage', fontsize=14, fontweight='bold')
        ax.set_xlabel('Data preparation stage', fontsize=12)
        ax.set_ylabel('Test RMSE (rides per hour, lower is better)', fontsize=12)
        ax.set_ylim(0, max(scores) * 1.15)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)

        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

        plt.xticks(rotation=15)
        plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()


def main():
    print()
    print()
    print('Loading data...')
    df=pd.read_csv('data/raw/store_sharing.csv').drop(columns=['cnt','timestamp'])
    df1=pd.read_csv('data/processed/encoded_ride_data.csv').drop(columns=['timestamp'])
    df2=pd.read_csv('data/processed/scaled_ride_data.csv').drop(columns=['timestamp'])
    df3=pd.read_csv('data/processed/fully_engineered_ride_data.csv')
    y=pd.read_csv('data/processed/ride_data_target.csv').squeeze()

    pipeline=SplitTrainTest(df,df1,df2,df3,y)
    rmse_dict=pipeline.run_pipe(df,df1,df2,df3,y)
    pipeline.plot_rmse(rmse_dict)

    print('Done!')
    

if __name__ == '__main__':
   main()