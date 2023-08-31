
from src.models.gpt4all_model import MyGPT4ALL
from src.knowledge_base.knowledgebase import MyKnowledgeBase
from src.knowledge_base.knowledgebase import (
    DOCUMENT_SOURCE_DIRECTORY
)
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms import Replicate
import os


def main():
   
    os.environ["REPLICATE_API_TOKEN"] = "dwejfwe"
    llm = Replicate(
    model="replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf",
    input={"temperature": 0.2,
           "max_length": 150,
           "top_p": 1},
)

    chat_model = MyGPT4ALL(
        model_folder_path=r'C:\Users\Jathushan\Documents\DS project\end-to-end-llm-main\src\models',
        model_name='llama-2-7b-chat.ggmlv3.q4_0.bin',
        allow_download=False,
        allow_streaming=True,
    )

    kb = MyKnowledgeBase(
        pdf_source_folder_path=DOCUMENT_SOURCE_DIRECTORY,vector_db= 'vect'
    )

    kb.initiate_document_injetion_pipeline()


    # get the retriver object from the vector db 

    retriever = kb.return_retriever_from_persistant_vector_db()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    custom_prompt_template = """
I want you to act as a hospital chatbot. My request is "{query}."
"""

    
    
    CUSTOM_QUESTION_PROMPT = PromptTemplate(input_variables = ['query'], template=custom_prompt_template)



    qa_chain = ConversationalRetrievalChain.from_llm(
    llm = chat_model,
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=False, 
    return_generated_question = False,
    verbose=True,
    memory = memory
    )
    vectdb = kb.get_vector_db()
    chat_history = []

    while True:
        query = input("User: ")
        if query == 'exit':
            break

        result = qa_chain({"question": CUSTOM_QUESTION_PROMPT.format(query=query), "chat_history": chat_history})
        print(result["answer"])
        chat_history.extend([(query, result["answer"])])
        # answer = result['result']

        # print(f'BOT: {answer}')

    # print("#"* 30, "Sources", "#"* 30)
    # for document in docs:
    #     print("\n> SOURCE: " + document.metadata["source"] + ":")
    #     print(document.page_content)
    # print("#"* 30, "Sources", "#"* 30)





    # while True:
    #     query = input('Enter your Query: ')
    #     if query == 'exit':
    #         break
    #     # use hydra to fill the **kwargs
    #     response = chat_model(
    #         query,
    #         n_predict=128,
    #         temp=1,
    #         top_p=0.01,
    #         top_k=40,
    #         n_batch=8,
    #         repeat_last_n=64,
    #         repeat_penalty=1.18,
    #         max_tokens=100,
    #     )
    #     print()


if __name__ == '__main__':
    main()