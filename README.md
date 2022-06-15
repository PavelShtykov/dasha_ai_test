# Тестовое задание

- Скрипты вида `partN_*.py` - основные скрипты, в которых происходит вся обработка датасетов и обучение моделей.
- [`pipeline.sh`](pipeline.sh) - bash-скрипт запускающий весь процесс анализа и обучения (Осторожно! Он перезаписывает папку с промежуточными результатами [`res`](res))
- [`requirements.txt`](requirements.txt)
- Модель обучалась на датасете [`res/part2/union_train.json`](res/part2/union_train.json). Он представлен в неком общем виде. Для каждой модели датасет переформатировался под требуемы формат. Переформатированные датасеты можно найти в [`res/part3/flair`](res/part3/flair) и в [`res/part3/w2ner`](res/part3/w2ner)
- json со статистикой (из 3-го пункта задания) - [`res/part2/union_stat.json`](res/part2/union_stat.json)
- Веса обученных моделей можно скачать с [гугл диска](https://drive.google.com/drive/folders/1nAVDnD9hnvO4lwH8vQI6zisgraago3ym?usp=sharing). [`eval_flair.py`](eval_flair.py) - скрипт с примером применения обученной [flair](https://github.com/flairNLP/flair) модели. Для W2NER модели аналогичного скрипта сделать не успел.
- [Отчет](report.pdf) о проделанной работе
- [`data`](data) - папка с исходными датасетами
- [`w2ner`](w2ner) - папка с моделью [W2NER](https://github.com/ljynlp/W2NER)

