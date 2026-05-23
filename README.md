## Установка

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Данные

### AG News

```
python -m src.train_classifier --config configs/lstm_baseline.yaml --ag_news_path data/train.csv
```
### text8

```
python -m src.train_word2vec --vector_size 100 --epochs 5 --workers 4 --output artifacts/text8_word2vec.model
```

## Базовое обучение

```
python -m src.train_word2vec --vector_size 100 --epochs 5 --workers 4 --output artifacts/text8_word2vec.model
python -m src.train_classifier --config configs/lstm_baseline.yaml --run_name lstm_baseline --ag_news_path /path/to/train.csv
```

GRU-вариант:

```
python -m src.train_classifier --config configs/gru_baseline.yaml --run_name gru_baseline --ag_news_path /path/to/train.csv
```

## Исследование гиперпараметров

```
python -m src.run_experiments --experiments configs/experiments.yaml --ag_news_path /path/to/train.csv
```

Итоговая таблица будет сохранена в:

```
outputs/experiments_summary.csv
```

## Графики обучения

```
python -m src.plot_history --history outputs/lstm_baseline/history.csv
```

## Предсказание по сохранённой модели

```
python -m src.predict \
  --checkpoint outputs/lstm_baseline/best_model.pt \
  --text "..."
```