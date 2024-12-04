# MLP를 사용한 LA 에어비앤비 숙소 가격 예측 모델 

## 사용한 프로그래밍 언어와 라이브러리
python 3.7

PyTorch, Pandas, Sklearn, Numpy, Matplot

## 사용한 데이터

LA 에어비앤비 숙소 데이터셋: [kaggle](https://www.kaggle.com/datasets/oscarbatiz/los-angeles-airbnb-listings)

## 예측 모델 생성 과정 및 결과

### 1. 데이터 정보 확인하기
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 45533 entries, 0 to 45532
Data columns (total 25 columns):
 #   Column                        Non-Null Count  Dtype  
---  ------                        --------------  -----  
 0   id                            45533 non-null  int64  
 1   name                          45532 non-null  object 
 2   host_id                       45533 non-null  int64  
 3   host_name                     45531 non-null  object 
 4   host_since                    45531 non-null  object 
 5   host_response_time            35445 non-null  object 
 6   host_response_rate            35445 non-null  float64
 7   host_is_superhost             44281 non-null  object 
 8   neighbourhood_cleansed        45533 non-null  object 
 9   neighbourhood_group_cleansed  45533 non-null  object 
 10  latitude                      45533 non-null  float64
 11  longitude                     45533 non-null  float64
 12  property_type                 45533 non-null  object 
 13  room_type                     45533 non-null  object 
 14  accommodates                  45533 non-null  int64  
 15  bathrooms                     37294 non-null  float64
 16  bedrooms                      42494 non-null  float64
 17  beds                          37199 non-null  float64
 18  price                         37296 non-null  float64
 19  minimum_nights                45533 non-null  int64  
 20  availability_365              45533 non-null  int64  
 21  number_of_reviews             45533 non-null  int64  
 22  review_scores_rating          33387 non-null  float64
 23  license                       12803 non-null  object 
 24  instant_bookable              45533 non-null  object 
dtypes: float64(8), int64(6), object(11)
```
25개의 칼럼으로 구성된 45533개의 데이터

```python
df.head()
```

|index|id|name|host\_id|host\_name|host\_since|host\_response\_time|host\_response\_rate|host\_is\_superhost|neighbourhood\_cleansed|neighbourhood\_group\_cleansed|latitude|longitude|property\_type|room\_type|accommodates|bathrooms|bedrooms|beds|price|minimum\_nights|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|670339032744709144|Westwood lovely three bedrooms three bathrooms|4780152|Moon|20/01/13|within a few hours|0\.96|f|West Los Angeles|City of Los Angeles|34\.04966|-118\.43555|Entire condo|Entire home/apt|6|3\.0|3\.0|3\.0|399\.0|30|
|1|37014494|Spanish style lower duplex near Beverly Hills|278288178|Ida|22/07/19|NaN|NaN|f|Beverlywood|City of Los Angeles|34\.04841|-118\.38751|Entire rental unit|Entire home/apt|2|NaN|2\.0|NaN|NaN|30|
|2|1024835174766068422|Charming Beverly Hills Home|513813179|Tiana|08/05/23|within a day|0\.6|f|Beverly Hills|Other Cities|34\.07058324822541|-118\.3907421|Entire home|Entire home/apt|6|3\.0|3\.0|3\.0|434\.0|30|
|3|850744632375448560|Tianpu's warm room with bathroom|432956623|Dan|22/11/21|a few days or more|0\.2|f|Temple City|Other Cities|34\.10932862747781|-118\.0730982|Private room in home|Private room|2|1\.0|1\.0|1\.0|49\.0|1|
|4|953950676345326970|Santa Monica apt, free parking, steps to the beach|528669205|Farkhat|29/07/23|within an hour|1\.0|t|Santa Monica|Other Cities|34\.01559|-118\.49408|Entire rental unit|Entire home/apt|2|1\.0|0\.0|1\.0|231\.0|5|

```python
df.isnull().sum()
```
<img width="265" alt="스크린샷 2024-12-05 오전 2 13 26" src="https://github.com/user-attachments/assets/719933b3-3021-4986-ac9a-2fb01efbb453">

**결측치 제거 후 남은 데이터 수 확인**
```python
df.dropna(inplace=True)
df.shape
```
```
(8366, 25)
```

**25개의 칼럼들 중 사용할 칼럼은 15개로, 입력 및 출력 데이터는 다음과 같다.**

- host_response_time: 호스트가 게스트의 문의에 응답하는 전형적인 시간.
- host_response_rate: 호스트가 응답한 게스트 문의의 비율.
- host_is_superhost: 호스트가 Superhost인지 여부 (True/False).
- neighbourhood_group_cleansed: 숙소가
- room_type: 제공되는 객실 유형 (예: 전체 집/아파트, 개인실, 공유실).
- accommodates: 속성이 수용할 수 있는 최대 게스트 수.
- bathrooms: 속성에 있는 욕실 수.
- bedrooms: 속성에 있는 침실 수.
- beds: 속성에 있는 침대 수.
- price: 예약에 필요한 최소 숙박일수를 기준으로 한 총 가격. (✅ 출력 데이터)
- minimum_nights: 예약에 필요한 최소 숙박일 수.
- availability_365: 다음 365일 동안 속성이 예약 가능한 일수.
- number_of_reviews: 속성이 받은 총 리뷰 수.
- review_scores_rating: 게스트 리뷰를 바탕으로 한 평균 평점 (최대 값은 5).
- instant_bookable: 게스트가 즉시 예약할 수 있는지 여부 (True/False).
출력 데이터 외의 데이터들은 입력 데이터로 사용한다.

### 2. 데이터 전처리
(1) 결측치 제거
(2) 불필요한 칼럼 제거
(3) 데이터셋을 torch.tensor로 변환하기 위해 범주형 데이터들을 매핑하여 정수화시킨다.
```python
# utils.py 코드 일부
mapping_room = {
        'Entire home/apt': 0,
        'Private room': 1,
        'Shared room': 2,
        'Hotel room': 3
    }
    df['room_type'] = df['room_type'].map(mapping_room)
```

(4) Feature데이터와 Target 데이터로 나눈다.
```python
y = df['price']
    X = df.drop(columns=['price'])
```
(5) 정규화 및 스케일링
```python
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
```
(6) PyTorch Tensor로 변환 후 8:2 비율로 훈련 셋과 테스트 셋으로 나눈다.
```python
# Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32).view(-1, 1)

    # Split data into train and test sets (80:20 ratio)
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
```
```python
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)
# torch.Size([6692, 14])
# torch.Size([1674, 14])
# torch.Size([6692, 1])
# torch.Size([1674, 1])
```

### 3. 모델 생성 및 모델의 구조 
(1) MLP(Multi-Layer Perceptron)를 이용하여 Model을 구성한다.

```python
import torch
import torch.nn as nn

class PricePredictionModel(nn.Module):
    def __init__(self, input_dim):
        super(PricePredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)  # Single output for price prediction
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.output(x)

```
다층 구조로 2개의 은닉층과 1개의 출력층으로 구성한다. 활성화 함수는 LeakyReLU로 비선형성을 추가한다.
- fc1: 입력 데이터를 64차원으로 변환하는 Linear Layer
- fc2: 64차원에서 32차원으로 축소하는 Linear Layer
- output: 32차원에서 최종 출력값(1개)으로 변환하는 Linear Layer
- relu: 비선형 활성화 함수로 LeakyReLU를 사용하여 모델이 복잡한 패턴을 학습할 수 있도록 한다.

### 4. 모델 학습 및 저장
```python
    # 모델 초기화
    input_dim = X_train.shape[1]
    model = PricePredictionModel(input_dim)

    # # 옵티마이저 및 손실 함수 정의
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    for epoch in range(100):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(
            f'Epoch {epoch + 1} | Train Loss: {train_loss / len(train_loader)} | Val Loss: {val_loss / len(val_loader)}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1} with validation loss: {val_loss:.4f}")
```
(1) 학습 단계: 데이터 로더에서 배치를 가져와 모델을 학습시킨다. 손실을 역전파하여 가중치를 업데이트한다.

(2) 검증 단계: 모델을 평가 모드로 전환한 뒤, 검증 데이터에서 손실을 계산한다. 

(3) 손실 기록 및 출력: 각 에포크의 학습 및 검증 손실을 저장하고, 출력한다.

(4) 최적 모델 저장: 검증 손실이 최소화 되는 순간의 모델을 저장한다.

<img width="642" alt="스크린샷 2024-12-05 오전 2 34 21" src="https://github.com/user-attachments/assets/2e573e2b-91a1-4911-9673-457f4a8c50c9">

(5) 결과 시각화

<img width="707" alt="image" src="https://github.com/user-attachments/assets/285254ed-b2e8-4c24-afe9-cfe0bb1060d1">



### 5. 하이퍼 파라미터 변경하여 낮은 Loss 찾기

<img width="1244" alt="스크린샷 2024-12-05 오전 2 35 59" src="https://github.com/user-attachments/assets/fdc976d3-8faf-4837-833d-940011bbf929">

LeakyReLU, batch_size = 64, epochs = 100 사용

### 6. 예측 (predict)

(1) 데이터셋을 불러와 데이터 전처리 및 데이터 셋 분할 과정을 거친다.

(2) 모델을 로드한다. 

```python
input_size = X_train.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PricePredictionModel(input_size)
save_path = "best_model_4.pth"
model.load_state_dict(torch.load(save_path, map_location=device))
model.to(device)
model.eval()
```
```
PricePredictionModel(
  (fc1): Linear(in_features=14, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=32, bias=True)
  (output): Linear(in_features=32, out_features=1, bias=True)
  (relu): LeakyReLU(negative_slope=0.01)
)
```

(3) 예측을 수행한다. 

```python
# 예측 수행
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
with torch.no_grad():
    predictions = model(X_val_tensor).cpu().numpy()
```

(4) MSE, RMSE를 계산 후 시각화한다.

```python
# 실제 값으로 변환
actual_predictions = scaler.inverse_transform(predictions)
actual_y_val = scaler.inverse_transform(y_val.reshape(-1, 1))

# MSE, RMSE 계산
mse = mean_squared_error(actual_y_val, actual_predictions)
rmse = np.sqrt(mse)
print(f"MSE: {mse}, RMSE: {rmse}")
```

```
MSE: 162678.28116705644, RMSE: 403.33395736914645
```

<img width="499" alt="image" src="https://github.com/user-attachments/assets/a69a1f45-3d6b-4c2d-8b69-0f627dc5d6f4">

### 7. 결과 정리

1. 일반적인 분포: 대부분의 점들이 그래프의 왼쪽 아래 구석에 밀집해 있다. 이는 예측 값이 실제 값에 비해 낮게 분포하고 있다는 것을 의미한다. 대부분의 price가 상대적으로 낮게 예측된 것으로 보인다.
2. 벗어난 데이터 포인트: 그래프의 오른쪽 상단에 몇 개의 데이터 포인트가 떨어져 있는데, 이는 **이상치(outliers)** 일 가능성이 커 보인다. 예측 값이 매우 높은 가격들이 실제 값보다 더 크게 예측된 경우인데, 이는 모델이 일부 높은 가격을 과대 예측한 결과인 것으로 보인다.

💡 개선할 수 있는 점

1. 이상치 처리: 이상치를 처리하거나 모델을 개선하여 이러한 예측 오류를 줄일 수 있다고 생각한다.
2. 모델 튜닝: 더 복잡한 모델을 사용하거나 모델의 하이퍼파라미터 조정에 힘을 써 예측 성능을 높일 수 있다고 생각한다.
3. 변수 조정: 데이터에 포함된 변수들에 대한 재조정, 변환 등을 시도해 볼 수 있다고 생각한다.





