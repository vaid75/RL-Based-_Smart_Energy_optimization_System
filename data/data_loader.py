import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)

    
    df = df.dropna()
    df["time_of_day"] = df["time_of_day"].astype(int)
    df["current_load"] = df["current_load"].astype(float)
    df["price"] = df["price"].astype(float)

    return df.values
