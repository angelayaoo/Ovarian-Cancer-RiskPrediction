import pandas as pd
import os

def preprocess_all_datasets():
    TRAIN_PATH = "data/raw/chinese training dataset(All Raw Data).csv"
    WEST_CHINA_PATH = "data/raw/west_china.xlsx"
    JAPAN_PATH = "data/raw/japanese.xlsx"
    
    print("Processing raw training CSV...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_train.columns = [str(c).lower().strip() for c in df_train.columns]
    
    if 'type' in df_train.columns:
        df_train = df_train.rename(columns={'type': 'label'})
    
    df_train['label'] = df_train['label'].astype(str).str.lower().str.strip()
    df_train['label'] = df_train['label'].map({
        'healthy': 0, 'control': 0, 'normal': 0, 'benign': 0, 'ovarian cysts': 0, 'cysts': 0, '0': 0,
        'ovarian cancer': 1, 'cancer': 1, 'case': 1, 'patients': 1, 'tumor': 1, '1': 1
    }).fillna(0).astype(int)
    
    column_mapping = {}
    for col in df_train.columns:
        if 'ca125' in col or 'ca-125' in col: column_mapping[col] = 'ca125'
        elif 'he4' in col: column_mapping[col] = 'he4'
        elif col == 'age': column_mapping[col] = 'age'
    df_train = df_train.rename(columns=column_mapping)
    
    os.makedirs("data/processed", exist_ok=True)
    df_train.to_csv("data/processed/chinese_train_cleaned.csv", index=False)
    print("Success! Processed data saved.")

if __name__ == "__main__":
    preprocess_all_datasets()
