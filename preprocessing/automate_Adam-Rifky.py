# Import library
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump

# Load dataset
def load_data():
    print("Loading Data...")
    path_train = 'DatasetAirlinePassengerSatisfaction_raw/train.csv'
    path_test = 'DatasetAirlinePassengerSatisfaction_raw/test.csv'
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)

    # Menggabungkan train dan test
    df = [df_train, df_test]
    df = pd.concat(df)
    df = df.drop(columns=['Unnamed: 0'])
    df = df.reset_index(drop=True)
    print("Done Loading Data")
    return df

# Preprocessing dataset
def preprocess_data(df):
    print("Preprocessing Data...")
    # Menghapus kolom yang tidak diperlukan dan data duplikat
    df = df.drop(columns=['id', 'Arrival Delay in Minutes'])
    df = df.drop_duplicates()
    
    # Handling outlier untuk kolom Flight Distance.
    Q1 = df['Flight Distance'].quantile(0.25)
    Q3 = df['Flight Distance'].quantile(0.75)
    IQR = Q3 - Q1

    condition = ~((df['Flight Distance'] < (Q1 - 1.5 * IQR)) | (df['Flight Distance'] > (Q3 + 1.5 * IQR)))

    df = df[condition]

    # Handling outlier untuk kolom Departure Delay in Minutes.
    Q1 = df['Departure Delay in Minutes'].quantile(0.25)
    Q3 = df['Departure Delay in Minutes'].quantile(0.75)
    IQR = Q3 - Q1

    condition = ~((df['Departure Delay in Minutes'] < (Q1 - 1.5 * IQR)) | (df['Departure Delay in Minutes'] > (Q3 + 1.5 * IQR)))

    df = df[condition]

    # Binning
    bin = [0, 14, 24, 64, 100]
    label = ['Child', 'Youth', 'Adult', 'Senior']

    df['Age_Group'] = pd.cut(df['Age'], bins=bin, labels=label)

    df = df.drop(columns=['Age'])

    # Feature Encoding
    encoders = {}

    categorical = df.select_dtypes(include=['object', 'category']).columns

    for col in categorical:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Scaling
    numeric = df.select_dtypes(include=['number']).columns
    numeric = numeric.drop('satisfaction')

    scaler = StandardScaler()
    df[numeric] = scaler.fit_transform(df[numeric])

    # Joblib
    output_folder = 'DatasetAirlinePassengerSatisfaction_preprocessing'
    os.makedirs(output_folder, exist_ok=True)

    # Save scaler and encoders
    dump(scaler, os.path.join(output_folder, 'scaler.joblib'))
    dump(encoders, os.path.join(output_folder, 'encoders.joblib'))

    print("Done Preprocessing Data")
    return df

# Split and Save dataset
def save_data(df):
    print("Saving Data...")
    train_data, test_data = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['satisfaction']
    )
    
    # Output folder
    output_folder = 'DatasetAirlinePassengerSatisfaction_preprocessing'
    os.makedirs(output_folder, exist_ok=True)

    train_path = os.path.join(output_folder, 'data_train.csv')
    test_path = os.path.join(output_folder, 'data_test.csv')
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    print("Done Saving Data")

# Start processing
df = load_data()
df = preprocess_data(df)
save_data(df)