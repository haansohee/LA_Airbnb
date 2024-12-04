import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


def load_and_preprocess_data():
    # Load the dataset
    df = pd.read_csv('listings.csv')

    # Remove missing values
    df.dropna(inplace=True)
    # 불필요한 칼럼 제거
    columns_to_drop = ['id', 'name', 'host_id', 'host_since', 'host_name',
                       'latitude', 'longitude', 'license', 'neighbourhood_cleansed', 'property_type']
    df = df.drop(columns=columns_to_drop)

    mapping_group = {
        'City of Los Angeles': 0,
        'Other Cities': 1,
        'Unincorporated Areas': 2
    }
    df['neighbourhood_group_cleansed'] = df['neighbourhood_group_cleansed'].map(mapping_group)

    mapping_super = {
        'f': 0,
        't': 1
    }
    df['host_is_superhost'] = df['host_is_superhost'].map(mapping_super)
    df['instant_bookable'] = df['instant_bookable'].map(mapping_super)

    mapping_room = {
        'Entire home/apt': 0,
        'Private room': 1,
        'Shared room': 2,
        'Hotel room': 3
    }
    df['room_type'] = df['room_type'].map(mapping_room)

    mapping_time = {
        'within an hour': 0,
        'within a few hours': 1,
        'within a day': 3,
        'a few days or more': 4
    }
    df['host_response_time'] = df['host_response_time'].map(mapping_time)

    # Features and target variable
    y = df['price']
    X = df.drop(columns=['price'])

    # 정규화 및 스케일링
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32).view(-1, 1)

    # Split data into train and test sets (80:20 ratio)
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val, scaler

def get_dataloaders(X_train, X_val, y_train, y_val, batch_size):
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader