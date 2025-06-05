# Adult Income Classification Project

A machine learning project for predicting income levels using the UCI Adult Census Income dataset. This project implements multiple models (XGBoost, LightGBM, and PyTorch Neural Network) and provides a FastAPI web service for predictions.

## Project Structure
```
adult_income_classifier/
├── app/                    # FastAPI web application
│   └── main.py            # API endpoints and model serving (загрузка моделей, обработка запросов, возврат предсказаний)
├── data/                   # Data directory
│   ├── raw/               # Original dataset (исходные CSV/Excel файлы без изменений)
│   └── processed/         # Preprocessed data (очищенные и преобразованные данные готовые для обучения)
├── models/                 # Trained model files
│   ├── lightgbm_model.pkl  # Сохраненная модель LightGBM
│   ├── xgboost_model.pkl   # Сохраненная модель XGBoost
│   └── neural_net_model.pt # Сохраненная нейросетевая модель PyTorch
├── notebooks/             # Jupyter notebooks for analysis
│   └── exploration.ipynb  # Ноутбук с анализом данных и экспериментами (опционально)
├── src/                   # Source code
│   ├── data_balancing.py  # Скрипт для балансировки классов (SMOTE, undersampling и т.д.)          
│   ├── data_preprocessing.py # Скрипт для очистки и преобразования данных
│   ├── download_data.py    # Скрипт для загрузки исходного датасета
│   ├── train_models.py     # Основной скрипт обучения моделей
│   ├── training_models_for_final_comparison.py # Финалное сравнение моделей с подбором гиперпараметров
│   └── visualize_data.py  # Скрипты для генерации графиков и визуализаций            
├── requirements.txt       # Project dependencies (список всех зависимостей Python)
├── report.md             # Project report and analysis (анализ результатов, метрик, выводы)
└── README.md             # Основной файл с описанием проекта, инструкциями по установке и запуску
```

## Features
- Multiple ML models (XGBoost, LightGBM, PyTorch Neural Network)
- Model interpretation using SHAP
- FastAPI web service for predictions
- Data loading from external sources
- Comprehensive model comparison
- Detailed documentation

## Requirements
- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/adult_income_classifier.git
cd adult_income_classifier
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing
The raw data is automatically processed when running the training script. The preprocessing includes:
- Handling missing values
- Feature encoding
- Feature scaling
- Class imbalance handling

### Model Training
To train all models:
```bash
python src/train.py
```

This will:
- Preprocess the data
- Train XGBoost, LightGBM, and Neural Network models
- Perform hyperparameter tuning
- Save the best models
- Generate performance metrics

### Model Performance
| Model    | Accuracy | Precision | Recall | F1-Score |
|----------|----------|-----------|---------|-----------|
| XGBoost  | 0.876    | 0.853     | 0.908   | 0.880    |
| LightGBM | 0.882    | 0.862     | 0.911   | 0.886    |
| NeuralNet| 0.858    | 0.825     | 0.909   | 0.865    |

### Web API
Start the FastAPI server:
```bash
python app/main.py
```

The API will be available at `http://localhost:8000` with the following endpoints:
- `GET /`: Health check
- `POST /predict`: Make predictions
- `GET /load_data`: Load data from external source

API Documentation is available at `http://localhost:8000/docs`

### Making Predictions
Using curl:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [39, 1, 2, 13, 1, 2, 1, 1, 1, 2174, 0, 40, 1, 0]}'
```

Using Python:
```python
import requests
import json

response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [39, 1, 2, 13, 1, 2, 1, 1, 1, 2174, 0, 40, 1, 0]}
)
print(json.dumps(response.json(), indent=2))
```

## Model Details

### XGBoost
- Gradient boosting implementation
- Handles missing values automatically
- Good for tabular data
- Fast training and inference

### LightGBM
- Light Gradient Boosting Machine
- Leaf-wise tree growth
- Faster training than XGBoost
- Good for large datasets

### Neural Network (PyTorch)
- Simple MLP architecture
- One hidden layer (64 neurons)
- Dropout for regularization
- ReLU activation
- Binary cross-entropy loss

## Model Interpretation
The project uses SHAP (SHapley Additive exPlanations) for model interpretation:
- Feature importance analysis
- Individual prediction explanations
- Global model behavior insights

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- UCI Machine Learning Repository for the dataset
- FastAPI for the web framework
- SHAP for model interpretation
