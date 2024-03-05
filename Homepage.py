import streamlit as st
from streamlit_option_menu import option_menu
from app_utils import switch_page
from PIL import Image
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

im = Image.open("icon.png")
st.set_page_config(page_title="AI Interviewer", layout="centered", page_icon=im)

lan = st.selectbox("#### Language", ["English", "中文"])

if lan == "English":
    home_title = "AI Interviewer"
    home_introduction = "Welcome to AI Interviewer, empowering your interview preparation with generative AI."

    st.markdown(
        "<style>#MainMenu{visibility:hidden;}</style>",
        unsafe_allow_html=True
    )
    st.image(im, width=100)
    st.markdown(f"""# {home_title}""", unsafe_allow_html=True)
    st.markdown("""\n""")
    # st.markdown("#### Greetings")
    st.markdown("Welcome to AI Interviewer! 👏 AI Interviewer is your personal interviewer powered by generative AI that conducts mock interviews."
                "You can upload your resume and enter job descriptions, and AI Interviewer will ask you customized questions. Additionally, you can configure your own Interviewer!")
    st.markdown("""\n""")
    role = st.text_input("Enter your role")
    if role:
        st.markdown(f"Your role is {role}")

        llm = ChatGoogleGenerativeAI(
            model="gemini-pro")
        prompt = f"Provide the tech stack and responsibilities for the top 3 job recommendations based on the role: {role}. " + """
        For each job recommendation, list the required tech stack and associated responsibilities without giving any title or role name. 
        Ensure the information is detailed and precise.
        follwoing is for example purpose, have our response in this format:
        
        ]

        """

        analysis = llm.invoke(prompt)
        st.write(analysis.content)

        if 'tech_stack' not in st.session_state:
            st.session_state.tech_stack = ""
        if 'responsibilities' not in st.session_state:
            st.session_state.responsibilities = ""

        with st.form(key='input_form'):
            tech_stack = st.text_input("Enter preferred tech stack", key='tech_stack')
            responsibilities = st.text_input("Enter responsibilities", key='responsibilities')
            submit_button = st.form_submit_button(label='Submit')


        if submit_button:
            if tech_stack and responsibilities:
                llm2 = ChatGoogleGenerativeAI(model="gemini-pro")
                prompt = f"""Tech stack: {tech_stack}\nResponsibilities: {responsibilities} 
                create a job description based on tech stack, responsibilities and give tech stack, responsibilities and qualifications for job description
                example -
                Tech stack: all technical stack here
                Qualifications: all qualifications here
                Responsibilities: all responsibilities here
                """
                response = llm2.invoke(prompt)
                st.write(response.content)
                jd = response.content

                if jd:
                    # Save the jd into a json file
                    with open("job_description.json", "w") as f:
                        json.dump(jd, f)
                        st.success("Job description saved successfully!")

        # if tech_stack and responsibilities:
        #     llm2 = ChatGoogleGenerativeAI(model="gemini-pro")
        #     prompt = f"""Tech stack: {tech_stack}\nResponsibilities: {responsibilities} 
        #     create a job description based on tech stack, responsibilities and give tech stack, responsibilities and qualifications for job description
        #     example -
        #     Tech stack: all technical stack here
        #     Qualifications: all qualifications here
        #     Responsibilities: all responsibilities here
        #     """
        #     response = llm2.invoke(prompt)
        #     st.write(response.content)
        #     jd = response.content

        #     if jd:
        #         # Save the jd into a json file
        #         with open("job_description.json", "w") as f:
        #             json.dump(jd, f)
        #         st.success("Job description saved successfully!")
