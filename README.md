# Машинное обучение — лабораторные работы

Дисциплина «Основы машинного обучения», 6 семестр.
Каждая работа состоит из **ноутбука** с реализацией (код + выводы + графики)
и **отчёта** в `.docx`/`.pdf`.

> **Как смотреть:** колонка **Ноутбук** открывается прямо в браузере на GitHub
> (со всеми ячейками и результатами); колонка **Отчёт** — готовый документ для проверки.
> Ссылка в колонке **#** ведёт в папку работы с условием задания (`README.md`).

## Лабораторные работы

| # | Тема | Ноутбук | Отчёт | Датасет |
|---|------|---------|-------|---------|
| [1](labs/lr-1/) | Работа с данными: очистка и отбор признаков | [main.ipynb](labs/lr-1/notebooks/main.ipynb) | [docx](labs/lr-1/report/lab.docx) · [pdf](labs/lr-1/report/lab.pdf) | ID_data_mass_18122012 |
| [2](labs/lr-2/) | Кластеризация | [main.ipynb](labs/lr-2/notebooks/main.ipynb) | [docx](labs/lr-2/report/lab.docx) | [Chemical Composition of Ceramic Samples](http://archive.ics.uci.edu/ml/datasets/Chemical+Composition+of+Ceramic+Samples) |
| [3](labs/lr-3/) | Ассоциативные правила | [main.ipynb](labs/lr-3/notebooks/main.ipynb) | [docx](labs/lr-3/report/lab.docx) · [pdf](labs/lr-3/report/lab.pdf) | [Online Retail](http://archive.ics.uci.edu/ml/datasets/Online+Retail) |
| [4](labs/lr-4/) | Машины опорных векторов в задаче определения P300 | [main.ipynb](labs/lr-4/notebooks/main.ipynb) | [docx](labs/lr-4/report/lab.docx) · [pdf](labs/lr-4/report/lab.pdf) | [MOABB / P300](https://gitlab.com/impulse-neiry/posts/-/blob/master/post01_simple_p300/post01ru_simple_p300.ipynb) |
| [5](labs/lr-5/) | Градиентный бустинг в распознавании человеческой активности | [main.ipynb](labs/lr-5/notebooks/main.ipynb) | [docx](labs/lr-5/report/lab.docx) · [pdf](labs/lr-5/report/lab.pdf) | [HAR Using Smartphones](http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) |
| [6](labs/lr-6/) | Random Forest: классификация поверхности по данным IMU-сенсоров | [main.ipynb](labs/lr-6/notebooks/main.ipynb) | [docx](labs/lr-6/report/lab.docx) · [pdf](labs/lr-6/report/lab.pdf) | [CareerCon 2019](https://www.kaggle.com/competitions/career-con-2019/overview) |
| [7](labs/lr-7/) | Многослойный персептрон: классификация грибов | [main.ipynb](labs/lr-7/notebooks/main.ipynb) | [docx](labs/lr-7/report/lab.docx) · [pdf](labs/lr-7/report/lab.pdf) | [Mushroom](https://archive.ics.uci.edu/ml/datasets/Mushroom) |

## Критерии оценивания

| Оценка | Требования |
|--------|-----------|
| **5** | Все алгоритмы реализованы **без** scikit-learn, PyTorch, TensorFlow, Keras. Разрешены: pandas, numpy, matplotlib |
| **4 и ниже** | Допускается использование ML-библиотек при условии, что студент может объяснить работу алгоритма |

## Структура работы

```
labs/lr-N/
├── README.md      # условие задания
├── notebooks/     # main.ipynb — реализация с выводами и графиками
├── report/        # lab.docx, lab.pdf — отчёт
├── assets/        # графики, сохранённые из ноутбука
├── src/           # вспомогательный код
└── data/          # датасеты (в git не хранятся)
```

## Общий код

- **`mlcore/`** — переиспользуемые модули: загрузка и предобработка табличных данных,
  метрики, визуализация. Используется в ноутбуках лабораторных.
