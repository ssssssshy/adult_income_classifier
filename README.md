# Income Classification Project
```bash
git clone https://github.com/ssssssshy/adult_income_classifier
```
```bash
python -m venv <имя_окружения>
```
```bash
venv\Scripts\activate
```
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
      📊 Сравнение моделей по метрикам:
           accuracy  precision    recall        f1
XGBoost    0.875745   0.852850  0.908188  0.879649
LightGBM   0.882366   0.861918  0.910616  0.885598
NeuralNet  0.857868   0.824755  0.908850  0.864763



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
