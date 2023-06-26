from langchain.prompts import PromptTemplate


template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT=PromptTemplate.from_template(template)


prompt_template = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say ""Я не уверен, что могу дать точный ответ по имеющейся документации. Добавьте, пожалуйста, документов или попробуйте перефразировать вопрос."". DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context. Respond in the language of the document.

{context}

Question: {question}
Helpful answer in markdown format:"""

QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

