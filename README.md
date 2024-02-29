# AI чатбот на питоне по чату с документацией в pdf

Это Python приложение для "Streamlit", которое позволяет загружать PDF файл(ы) и общаться с вашими документами.

## Что такое Streamlit?

Streamlit - это библиотека Python с открытым исходным кодом, которая упрощает создание и распространение пользовательских веб-приложений для машинного обучения c хорошим пользовательским интерфейсом. Всего за несколько минут можно создать и развернуть приложения для работы с данными в Streamlit Community Cloud бесплатно.

[Streamlit Docs](https://docs.streamlit.io/)

## Компоненты приложения

`streamlit.py` - запускаемый файл c возможностью использовать pinecone namespaces

`streamlit-pinecone-starter.py` - запускаемый файл без возможности использовать pinecone namespaces (для учетных записей pinecone, созданных в сентябре 2023 года, уже отсутствует возможность использования namespaces в рамках бесплатного тарифа: https://docs.pinecone.io/docs/namespaces)

Содержит функцию инъекции данных в базу данных pinecone и основную функцию с описанием логики семантического поиска по загружаемой базе знаний в pdf формате (можно заменить loaders и загружать файлы в любом другом формате). 
В коде используется Langchain фреймворк для создания LLM приложений RAG (Retrieval Augmented Generation) архитектуры.
Основные компоненты:
- LLM = gpt-3.5-turbo
- Embeddings = text-embedding-ada-002 (эмбеддинги openai)
- Vector DB = pinecone.io

prompts.py - файл содержит 2 шаблона промптов: 
1) template = для формирования из вопроса пользователя итогового вопроса с учетом контекста беседы (так называемый CONDENSE_QUESTION_PROMPT)
2) итоговый prompt_template в который входят: системный промпт + результаты поиска по векторной базе данных + вопрос пользователя с учетом истории общения
      
## Запуск в Streamlit Community Cloud

- Вы можете развернуть это приложение через Streamlit Community Cloud, следуя следующим инструкциям [docs](https://docs.streamlit.io/streamlit-community-cloud/get-started)
- Перед запуском приложения в Streamlit Community Cloud в App Settings -> Secrets необходимо будет задать следующие системные переменные (внутри " " необходимо вписать ваши значения ключей).
```
OPENAI_API_KEY=" "
PINECONE_API_KEY=" "
PINECONE_ENVIRONMENT=" "
PINECONE_INDEX_NAME=" "
PINECONE_NAMESPACE=" "
```
Если ваш тарифный план в pinecone не позволяет использовать namespaces, то задавать значение переменной `PINECONE_NAMESPACE` не нужно.
