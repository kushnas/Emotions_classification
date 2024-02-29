# MLops
Возмём датасет с текстами и шестью фундаментальными эмоциями, которые наиболее выражены в приведённом тексте. Каждая запись в этом наборе данных состоит из текстового сегмента, 
представляющего сообщение в Twitter, 
и соответствующей метки, указывающей на преобладающую передаваемую эмоцию. Эмоции подразделяются на шесть категорий: печаль (0), радость (1), любовь (2), гнев (3), страх (4) и удивление (5). 

Вот несколько примеров, где такая модель может быть полезна:
1. Социальные медиа платформы: Компании, занимающиеся социальными медиа, могут использовать модель для анализа эмоций в текстах пользователей.
   Это поможет понять, какие эмоции вызывают определенные контенты или события, что позволит улучшить стратегию контент-маркетинга и взаимодействия с аудиторией.
2. Компании в области клиентского обслуживания: Классификация эмоций в сообщениях от клиентов может помочь компаниям быстрее выявлять проблемы или недовольство клиентов,
    что позволит оперативно реагировать и улучшать качество обслуживания.
3. Финансовые институты: Анализ эмоций в текстах о финансовых событиях или компаниях может помочь инвесторам принимать более обоснованные решения,
    учитывая не только факты, но и эмоциональный настрой рынка.
4. Медицинские учреждения: Использование модели для анализа эмоций в текстах пациентов или отзывах о медицинских услугах может помочь улучшить качество обслуживания и поддержки пациентов.

Финансовый эффект от использования такой модели может быть разнообразным:
- Увеличение лояльности клиентов за счет более эмоционального взаимодействия.
- Снижение времени реакции на негативные эмоции клиентов, что поможет предотвратить потерю клиентов.
- Улучшение качества контента и рекламы на основе анализа эмоций аудитории.
- Более точное прогнозирование рыночных трендов и поведения потребителей.

Помимо вышеперечисленных отраслей, модель классификации эмоций может быть полезна в других областях,
где важно понимание эмоционального состояния людей на основе текстовых данных. 

Данные возьму с kaggle https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data

Буду использовать PyTorch CNN, так как это было успешно применено в этом ноутбуке https://www.kaggle.com/code/alkidiarete/emosion-cnn-roc-0-99
и следующие модули:
tensorflow.keras.preprocessing.text Tokenizer
tensorflow.keras.preprocessing.sequence pad_sequences
tensorflow.keras.models Sequential
tensorflow.keras.layers Embedding, Conv1D, GlobalMaxPooling1D, Dense
tensorflow.keras.callbacks EarlyStopping



