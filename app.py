import streamlit as st 
from streamlit_chat import message
from src.models.gpt4all_model import MyGPT4ALL
from src.knowledge_base.knowledgebase import MyKnowledgeBase
from src.knowledge_base.knowledgebase import (
    DOCUMENT_SOURCE_DIRECTORY
)

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms import Replicate
import os




# checkpoint = "LaMini-T5-738M"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# base_model = AutoModelForSeq2SeqLM.from_pretrained(
#     checkpoint,
#     device_map="auto",
#     torch_dtype = torch.float32,
#     from_tf=True
# )

    


@st.cache_resource(show_spinner=False)
def qa_llm():

    chat_model = MyGPT4ALL(
        model_folder_path=r'C:\Users\Jathushan\Documents\DS project\end-to-end-llm-main\src\models',
        model_name='llama-2-7b-chat.ggmlv3.q4_1.bin',
        allow_download=False,
        allow_streaming=True,
    )
    # os.environ["REPLICATE_API_TOKEN"] = "r8_6rXolartKZeUccahg23a7e9rrUMMUyU241P5V"
    # chat_model = Replicate(
    # model="replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf",
    # input={"temperature": 0.2,
    #        "max_length": 150,
    #        "top_p": 1},
    # )

    custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. At the start of standalone question add this 'I want you to act as a hospital chatbot'
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""


    kb = MyKnowledgeBase(
        pdf_source_folder_path=DOCUMENT_SOURCE_DIRECTORY,vector_db= 'vect'
    )
    kb.initiate_document_injetion_pipeline()
    retriever = kb.return_retriever_from_persistant_vector_db()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
    qa_chain = ConversationalRetrievalChain.from_llm(
    llm = chat_model,
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=False, 
    return_generated_question = False,
    condense_question_prompt=CUSTOM_QUESTION_PROMPT,
    verbose=True,
    memory = memory
    )
    return qa_chain


def process_answer(query):
    custom_prompt_template = """
I want you to act as a hospital chatbot. My request is "{query}."
""" 
    CUSTOM_QUESTION_PROMPT = PromptTemplate(input_variables = ['query'], template=custom_prompt_template)   

    qa_chain = qa_llm()
    result = qa_chain({"question": CUSTOM_QUESTION_PROMPT.format(query=query)})
    return result["answer"]

# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))

def main():
    with st.spinner(text="Initialising....."):
        qa_llm()
    
    st.title("🏥 HealthBot 🤖")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Please Enter Your Medical Inquiry or Question"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner('please wait...'):
            response = process_answer(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()