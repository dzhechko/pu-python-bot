# создаем простое streamlit приложение для общения с вашим pdf

import streamlit as st
import tempfile
import os
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
import pinecone
from langchain.chains.question_answering import load_qa_chain
from prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains import LLMChain
from streamlit_chat import message


# использовать системные переменные из облака streamlit (secrets)
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_environment = st.secrets["PINECONE_ENVIRONMENT"]
pinecone_index = st.secrets["PINECONE_INDEX_NAME"]
pinecone_namespace = st.secrets["PINECONE_NAMESPACE"]


def ingest_docs(temp_dir: str = tempfile.gettempdir()):
    """
    Инъекция ваших pdf файлов в Pinecone
    """
    try:
        # выдать ошибку, если каких-то переменных не хватает
        if not openai_api_key or not pinecone_api_key or not pinecone_environment or not pinecone_index or not pinecone_namespace:
            raise ValueError(
                "Пожалуйста укажите необходимый набор переменных окружения")

        # загрузить PDF файлы из временной директории
        loader = DirectoryLoader(
            temp_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, recursive=True
        )
        documents = loader.load()

        # разбиваем документы на блоки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100)
        documents = text_splitter.split_documents(documents)

        # подключаемся к базе данных Pinecone, используя наши ключи
        pinecone.init(
            api_key=pinecone_api_key, environment=pinecone_environment)

        # инициируем процедуру превращения блоков текста в Embeddings через OpenAI API, используя API ключ доступа
        embeddings = OpenAIEmbeddings(
            model='text-embedding-ada-002', openai_api_key=openai_api_key)

        # добавляем "документы" (embeddings) в векторную базу данных Pinecone
        Pinecone.from_documents(
            documents, embeddings, index_name=pinecone_index, namespace=pinecone_namespace)
    except Exception as e:
        st.error(f"Возникла ошибка при добавлении ваших файлов: {str(e)}")


# это основная функция, которая запускает приложение streamlit
def main():
    # Указываем название и заголовок Streamlit приложения
    st.title('Чат с вашими PDF файлами')
    st.write('Загружайте свои PDF-файлы и задавайте вопросы по ним. Если вы уже загрузили свои файлы, то переходите к чату ниже.')

    # Выводим предупреждение, если пользователь не указал свои учетные данные
    if not openai_api_key or not pinecone_api_key or not pinecone_environment or not pinecone_index or not pinecone_namespace:
        st.warning(
            "Пожалуйста, задайте свои учетные данные в streamlit secrets для запуска этого приложения.")

    # Загрузка pdf файлов
    uploaded_files = st.file_uploader(
        "После загрузки файлов в формате pdf начнется их инъекция в векторную БД.", accept_multiple_files=True, type=['pdf'])

    # если файлы загружены, сохраняем их во временную папку и потом заносим в vectorstore
    if uploaded_files:
        # создаем временную папку и сохраняем в ней загруженные файлы
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    file_name = uploaded_file.name
                    # сохраняем файл во временную папку
                    with open(os.path.join(temp_dir, file_name), "wb") as f:
                        f.write(uploaded_file.read())
                # отображение спиннера во время инъекции файлов
                with st.spinner("Добавление ваших файлов в базу ..."):
                    ingest_docs(temp_dir)
                    st.success("Ваш(и) файл(ы) успешно принят(ы)")
                    st.session_state['ready'] = True
        except Exception as e:
            st.error(
                f"При загрузке ваших файлов произошла ошибка: {str(e)}")

    # Логика обработки сообщений от пользователей
    # инициализировать историю чата, если ее пока нет 
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # инициализировать состояние готовности, если его пока нет
    if 'ready' not in st.session_state:
        st.session_state['ready'] = True

    if st.session_state['ready']:

        # инициализировать Pinecone, используя учетные данные
        pinecone.init(
            api_key=pinecone_api_key, environment=pinecone_environment)

        # инициализировать модели OpenAIEmbeddings и чата
        embeddings = OpenAIEmbeddings(
            model='text-embedding-ada-002', openai_api_key=openai_api_key)

        # можете измените модель на gpt-4, если у вас есть доступ к соответствующему api
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0,
                         openai_api_key=openai_api_key, verbose=True)

        # инициализация памяти
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # инициализация retrival chain - цепочки поиска
        vectorstore = Pinecone.from_existing_index(
            index_name=pinecone_index, embedding=embeddings, text_key='text', namespace=pinecone_namespace)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        question_generator = LLMChain(
            llm=llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=True)
        doc_chain = load_qa_chain(
            llm, chain_type="stuff", prompt=QA_PROMPT, verbose=True)

        qa = ConversationalRetrievalChain(
            retriever=retriever, question_generator=question_generator, combine_docs_chain=doc_chain, verbose=True, memory=memory)

        if 'generated' not in st.session_state:
            st.session_state['generated'] = [
                "Что бы вы хотели узнать о документе?"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Привет!"]

        # контейнер для истории чата
        response_container = st.container()

        # контейнер для текстового поля
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input(
                    "Вопрос:", placeholder="О чем этот документ?", key='input')
                submit_button = st.form_submit_button(label='Отправить')

            if submit_button and user_input:
                # отобразить загрузочный "волчок"
                with st.spinner("Думаю..."):
                    print("История чата: ", st.session_state['chat_history'])
                    output = qa(
                        {"question": user_input})
                    print(output)
                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(output['answer'])

                    # # обновляем историю чата с помощью вопроса пользователя и ответа от бота
                    st.session_state['chat_history'].append(
                        {"вопрос": user_input, "ответ": output['answer']})

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(
                        i) + '_user')
                    message(st.session_state["generated"][i], key=str(
                        i))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.write(f"Что-то пошло не так. Пожалуйста, попробуйте еще раз. {str(e)}")
