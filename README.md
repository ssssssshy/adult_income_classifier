# Income Classification Project

## Описание
Реализация моделей для бинарной классификации дохода (UCI Adult dataset).  
Используемые модели: 
- XGBoost
- LightGBM 
- Простая нейронная сеть на PyTorch

## Требования
- Python 3.8+
- Библиотеки:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - lightgbm
  - torch (PyTorch)

### Установка зависимостей
```bash
pip install -r requirements.txt
```

### Инструкция по запуску
- Подготовка данных
Обработанные данные должны находиться в data/processed/.

- Обучение моделей
Запуск обучения и подбора гиперпараметров:

```bash
python training_models_for_final_comparison.py
```
- Оценка результатов
   
    - Метрики по моделям:
        Выводятся в консоль



# Загрузка модели
```bash
model = torch.load('models/neural_net_model.pt')
model.eval()
```

# Получение предсказаний
```bash
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
preds = model(X_test_tensor).detach().numpy()
```

# Сохранение моделей
XGBoost: models/xgboost_model.pkl

LightGBM: models/lightgbm_model.pkl

Нейросеть: models/neural_net_model.pt
