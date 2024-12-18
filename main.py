from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd
import numpy as np

# Загрузка модели и LabelEncoders
model = load("xgb_model_compressed.joblib")
label_encoders = load("label_encoders.joblib")  # Предполагается, что LabelEncoders были сохранены

# Создание приложения FastAPI
app = FastAPI()


# Модель данных для входных параметров
class FlightData(BaseModel):
    MONTH: int
    DAY_OF_MONTH: int
    DAY_OF_WEEK: int
    OP_UNIQUE_CARRIER: str
    TAIL_NUM: str
    DEST: str
    CRS_ELAPSED_TIME: float
    DISTANCE: float
    CRS_DEP_M: int
    DEP_TIME_M: int
    CRS_ARR_M: int
    sch_dep: int
    sch_arr: int
    Temperature: float
    Humidity: float
    Wind: str
    Wind_Speed: float
    Wind_Gust: float
    Pressure: float
    Condition: str
    TAXI_OUT: int
    Dew_Point: float


@app.post("/predict/")
async def predict_delay(data: FlightData):
    # Преобразование входных данных в DataFrame
    input_data = pd.DataFrame([data.dict(by_alias=True)])

    # Переименование колонок для соответствия модели
    input_data.rename(columns={
        "Dew_Point": "Dew Point",
        "Wind_Speed": "Wind Speed",
        "Wind_Gust": "Wind Gust"
    }, inplace=True)

    # Загрузка ожидаемых фичей и их порядка из модели
    expected_features = model.get_booster().feature_names

    # Перестановка колонок в правильном порядке
    input_data = input_data[expected_features]

    # Кодирование категориальных признаков
    categorical_columns = ["OP_UNIQUE_CARRIER", "TAIL_NUM", "DEST", "Wind", "Condition"]
    for col in categorical_columns:
        if col in input_data:
            input_data[col] = input_data[col].map(
                lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else 0
            )

    # Преобразование данных в числовые типы
    input_data = input_data.astype(np.float32)

    # Предсказание задержки
    prediction = model.predict(input_data)

    # Возврат результата
    return {"predicted_delay": float(prediction[0])}