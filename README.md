# Машинное обучение — Лабораторные работы

## Критерии оценивания

| Оценка | Требования |
|--------|-----------|
| **5** | Все алгоритмы реализованы **без** scikit-learn, PyTorch, TensorFlow, Keras. Разрешены: pandas, numpy, matplotlib |
| **4 и ниже** | Допускается использование ML-библиотек при условии, что студент может объяснить работу алгоритма |

## Лабораторные работы

| # | Тема | Алгоритмы / Методы | Датасет | Директория |
|---|------|---------------------|---------|------------|
| 1 | [Работа с данными](labs/lr-1/) | Gain ratio, корреляции, описательная статистика | ID_data_mass_18122012 | `labs/lr-1/` |
| 2 | [Кластеризация](labs/lr-2/) | K-means++, DBSCAN | [Chemical Composition of Ceramic Samples](http://archive.ics.uci.edu/ml/datasets/Chemical+Composition+of+Ceramic+Samples) | `labs/lr-2/` |
| 3 | [Ассоциативные правила](labs/lr-3/) | Apriori, FP-Growth | [Online Retail](http://archive.ics.uci.edu/ml/datasets/Online+Retail) | `labs/lr-3/` |
| 4 | [Машины опорных векторов](labs/lr-4/) | SVM (P300 BCI) | [MOABB](https://gitlab.com/impulse-neiry/posts/-/blob/master/post01_simple_p300/post01ru_simple_p300.ipynb) | `labs/lr-4/` |
| 5 | [Градиентный бустинг](labs/lr-5/) | Gradient Boosting | [HAR Using Smartphones](http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) | `labs/lr-5/` |
| 6 | [Random Forest](labs/lr-6/) | Random Forest (не бинарное дерево) | [Career Con 2019](https://www.kaggle.com/competitions/career-con-2019/overview) | `labs/lr-6/` |
| 7 | [MLP](labs/lr-7/) | Многослойный персептрон | [Mushroom](https://archive.ics.uci.edu/ml/datasets/Mushroom) | `labs/lr-7/` |

## Структура каждой лабораторной

```
labs/lr-N/
├── assets/       # Графики и визуализации
├── data/         # Датасеты (игнорируются git)
├── notebooks/    # Jupyter-ноутбуки
├── report/       # Отчёт
├── src/          # Исходный код
└── README.md     # Задание
```

## Общие библиотеки

- **`mlcore/`** — переиспользуемые модули для работы с табличными данными (загрузка, предобработка, визуализация)
