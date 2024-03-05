import streamlit as st
from streamlit_lottie import st_lottie
from typing import Literal
from dataclasses import dataclass
import json
import base64
from langchain.memory import ConversationBufferMemory

from langchain.chains import ConversationChain, RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import NLTKTextSplitter
from langchain.vectorstores import FAISS
import nltk
from prompts.prompts import templates
from langchain_google_genai import ChatGoogleGenerativeAI
import getpass
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCA4__JMC_ZIQ9xQegIj5LOMLhSSrn3pMw"



st.title("Professional Screen")

with open('job_description.json', 'r') as f:
    jd = json.load(f)

@dataclass
class Message:
    """class for keeping track of interview history."""
    origin: Literal["human", "ai"]
    message: str

def save_vector(text):
    """embeddings"""

    nltk.download('punkt')
    text_splitter = NLTKTextSplitter()
    texts = text_splitter.split_text(text)
     # Create emebeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    docsearch = FAISS.from_texts(texts, embeddings)
    return docsearch

def initialize_session_state_jd():
    """ initialize session states """
    if 'jd_docsearch' not in st.session_state:
        st.session_state.jd_docserch = save_vector(jd)
    if 'jd_retriever' not in st.session_state:
        st.session_state.jd_retriever = st.session_state.jd_docserch.as_retriever(search_type="similarity")
    if 'jd_chain_type_kwargs' not in st.session_state:
        Interview_Prompt = PromptTemplate(input_variables=["context", "question"],
                                          template=templates.jd_template)
        st.session_state.jd_chain_type_kwargs = {"prompt": Interview_Prompt}
    if 'jd_memory' not in st.session_state:
        st.session_state.jd_memory = ConversationBufferMemory()
    # interview history
    if "jd_history" not in st.session_state:
        st.session_state.jd_history = []
        st.session_state.jd_history.append(Message("ai",
                                                   "Hello, Welcome to the interview. I am your interviewer today. I will ask you professional questions regarding the job description you submitted."
                                                   "Please start by introducting a little bit about yourself. Note: The maximum length of your answer is 4097 tokens!"))
    # token count
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "jd_guideline" not in st.session_state:
        llm = ChatGoogleGenerativeAI(
        model="gemini-pro")
        st.session_state.jd_guideline = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type_kwargs=st.session_state.jd_chain_type_kwargs, chain_type='stuff',
            retriever=st.session_state.jd_retriever, memory = st.session_state.jd_memory).run("Create an interview guideline and prepare only one questions for each topic. Make sure the questions tests the technical knowledge")
    # llm chain and memory
    if "jd_screen" not in st.session_state:
        llm = ChatGoogleGenerativeAI(
        model="gemini-pro")
        PROMPT = PromptTemplate(
            input_variables=["history", "input"],
            template="""I want you to act as an interviewer strictly following the guideline in the current conversation.
                            Candidate has no idea what the guideline is.
                            Ask me questions and wait for my answers. Do not write explanations.
                            Ask question like a real person, only one question at a time.
                            Do not ask the same question.
                            Do not repeat the question.
                            Do ask follow-up questions if necessary. 
                            You name is GPTInterviewer.
                            I want you to only reply as an interviewer.
                            Do not write all the conversation at once.
                            If there is an error, point it out.

                            Current Conversation:
                            {history}

                            Candidate: {input}
                            AI: """)

        st.session_state.jd_screen = ConversationChain(prompt=PROMPT, llm=llm,
                                                           memory=st.session_state.jd_memory)
    if 'jd_feedback' not in st.session_state:
        llm = ChatGoogleGenerativeAI(
        model="gemini-pro")
        st.session_state.jd_feedback = ConversationChain(
            prompt=PromptTemplate(input_variables=["history", "input"], template=templates.feedback_template),
            llm=llm,
            memory=st.session_state.jd_memory,
        )

def answer_call_back():
    formatted_history = []
    for message in st.session_state.jd_history:
        if message.origin == "human":
            formatted_message = {"speaker": "user", "text": message.message}
        else:
            formatted_message = {"speaker": "assistant", "text": message.message}
        formatted_history.append(formatted_message)

    user_answer = st.session_state.get('answer', '')

    answer = st.session_state.jd_screen.run(input=user_answer, history=formatted_history)

    if user_answer:  
        st.session_state.jd_history.append(Message("human", user_answer))
    if answer:
        st.session_state.jd_history.append(Message("ai", answer))

    return answer


if jd:
    # initialize session states
    initialize_session_state_jd()
    #st.write(st.session_state.jd_guideline)
    credit_card_placeholder = st.empty()
    col1, col2 = st.columns(2)
    with col1:
        feedback = st.button("Get Interview Feedback")
    with col2:
        guideline = st.button("Show me interview guideline!")
    chat_placeholder = st.container()
    answer_placeholder = st.container()
    audio = None
    # if submit email adress, get interview feedback imediately
    if guideline:
        st.write(st.session_state.jd_guideline)
    if feedback:
        evaluation = st.session_state.jd_feedback.run("please give evalution regarding the interview")
        st.markdown(evaluation)
        st.download_button(label="Download Interview Feedback", data=evaluation, file_name="interview_feedback.txt")
        st.stop()
    else:
        with answer_placeholder:
            voice = 0
            if voice:
                print(voice)               
            else:
                answer = st.chat_input("Your answer")
            if answer:
                st.session_state['answer'] = answer
                audio = answer_call_back()
        with chat_placeholder:
            for answer in st.session_state.jd_history:
                if answer.origin == 'ai':
                    if audio:
                        with st.chat_message("assistant"):
                            st.write(answer.message)
                            st.write(audio)
                    else:
                        with st.chat_message("assistant"):
                            st.write(answer.message)
                else:
                    with st.chat_message("user"):
                        st.write(answer.message)

        credit_card_placeholder.caption(f"""
        Progress: {int(len(st.session_state.jd_history) * 10)}% completed.""")
else:
    st.info("Please submit a job description to start the interview.")
