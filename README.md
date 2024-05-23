# Emotions classification
Интерфейс клиентского взаимодействия выглядит следующим образом: клиент вводит текст в консоль, согласно скрипту clients_code.py, как результат получает ответ в виде одной из фундаментальных эмоций.
Эмоции подразделяются на шесть категорий: печаль (0), радость (1), любовь (2), гнев (3), страх (4) и удивление (5).

Возмём датасет с текстами и шестью фундаментальными эмоциями, которые наиболее выражены в приведённом тексте. Каждая запись в этом наборе данных состоит из текстового сегмента, 
представляющего сообщение в Twitter, 
и соответствующей метки, указывающей на преобладающую передаваемую эмоцию.  

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

Данные были использованы из источника: kaggle https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data

Буду использовать как опорный код реализацию модели в ноутбуке https://www.kaggle.com/code/alkidiarete/emosion-cnn-roc-0-99
Используемая в нём модель имеет следующий вид


![загруженное (4)](https://github.com/kushnas/Emotions_classification/assets/153232117/89c09b8c-c595-4a33-966a-399ac1b9942b)


