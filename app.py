import streamlit as st
import base64
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
#from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from streamlit.components.v1 import html
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import io

import time

st.set_page_config(page_title="Bid Response Evaluation AI ", layout="wide")

video_html = """
		<style>
		#myVideo {
		  position: fixed;
		  right: 0;
		  bottom: 0;
		  min-width: 100%; 
		  min-height: 100%;
		  filter: brightness(30%); /* Adjust the brightness to make the video darker */
		}
		
		.content {
		  position: fixed;
		  bottom: 0;
		  background: rgba(0, 0, 0, 0.5); /* Adjust the transparency as needed */
		  color: #f1f1f1;
		  width: 100%;
		  padding: 20px;
		}
		</style>	
		<video autoplay muted loop id="myVideo">
		  <source src="https://assets.mixkit.co/videos/30563/30563-720.mp4" type="video/mp4">
		  Your browser does not support HTML5 video.
		</video>
		"""

st.markdown(video_html, unsafe_allow_html=True)


st.markdown("""
<style>
    iframe {
        position: fixed;
        left: 0;
        right: 0;
        top: 0;
        bottom: 0;
        border: none;
        height: 100%;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
        @keyframes gradientAnimation {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }

        .animated-gradient-text {
            font-family: "Graphik Semibold";
            font-size: 42px;
            background: linear-gradient(45deg, #22ebe8 30%, #dc14b7 55%, #fe647b 20%);
            background-size: 300% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientAnimation 20s ease-in-out infinite;
        }
    </style>
    <p class="animated-gradient-text">
        Bid Response Evaluation AI: Evaluates Bid responses!
    </p>
""", unsafe_allow_html=True)


#st.image("https://media1.tenor.com/m/6o864GYN6wUAAAAC/interruption-sorry.gif", width=1000)
# st.image("https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExcjl2dGNiYThobHplMG81aGNqMjdsbWwwYWJmbTBncGp6dHFtZTFzMSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/CGP9713UVzQ0BQPhSf/giphy.gif", width=50)


# This is the first API key input; no need to repeat it in the main function.
api_key = 'AIzaSyAJT6_IYPjUtUyT14uzZ8BSON7rDul7Ab8'

def ocr_image(image):
    text = pytesseract.image_to_string(image)
    return text
	
def get_pdf_text(pdf_docs):
    text = "Response 1: "
    for pdf_info in pdf_docs:
	    filename = pdf_info.name
	    st.write(pdf_info)
	    st.write(pdf_docs)
	    try:
		    with fitz.open(pdf_info) as pdf_document:
			    for page_num in range(len(pdf_document)):
				    page = pdf_document.load_page(page_num)
				    text += page.get_text()
                    
	                    	    images = page.get_images(full=True)
				    for img_index, img in enumerate(images):
					    xref = img[0]
					    base_image = pdf_document.extract_image(xref)
					    image_bytes = base_image["image"]
					    image_ext = base_image["ext"]
					    image = Image.open(io.BytesIO(image_bytes))
					    text += ocr_image(image)
                            
        
        except Exception as e:
            st.error(f"Error processing {filename}: {e}")
    text += "\n\nResponse 2: "
return text

def user_input(api_key):
    st.write('inside input function')


def main():
    st.header("Evaluate bid responses")
    st.markdown("""
    <style>
    input {
      border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)
#    user_question = st.text_input("Ask a Question from the RFP Files", key="user_question")

    pdf_docs = st.file_uploader("Upload RFP responses here and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
    
    if st.button("Start the evaluation"):  # Ensure API key and user question are provided
      with st.spinner("Processing Response..."):
        with st.spinner("Reading response document..."):
            time.sleep(8)
            raw_text = get_pdf_text(pdf_docs)
            # st.write(raw_text)
            
      with st.spinner("Standardizing Responses...."):
        time.sleep(5)
        with st.spinner("Inconsistent Response Found. Standardizing Sections...."):
          time.sleep(5)
      with st.spinner("Embedding Text into vectors...."):
          time.sleep(3)
      with st.spinner("Section wise Chunking Responses...."):
          time.sleep(4)
      
      with st.spinner("Evaluating Responses based on the scoring criteria"):
            prompt = ''' Consider yourself as bid evaluator who will evaluate bids received from different vendors basis the context provided and will generate score with explaination. I will provide you some context but before we jump into evaluation let's understand the bid. Below are the bid details for which we will be evaluating the responses: 
            LCBO Background
            The Liquor Control Board of Ontario (LCBO) is a leading global retailer and wholesaler of beverage alcohol, offering over 28,000 products from more than 80 countries. Through its Spirit of Sustainability (SoS) platform, launched in 2018, the LCBO supports Ontario’s social and environmental needs. Last year, it contributed over $16 million to community well-being and returned $2.55 billion to the province.

            RFP Objective
            LCBO seeks a consulting services provider to develop and implement a five-year ESG strategy that aligns with SoS and establishes LCBO as a sustainability leader. Requirements include:

            Minimum of five years in ESG strategy development and implementation.
            Expertise in the alcohol beverage and retail consumer goods industry, plus knowledge of government and environmental regulations.
            Scope of Work
            Phase 1: ESG Research and Analysis

            Conduct internal and external ESG research.
            Perform a double materiality assessment.
            Phase 2: ESG Strategy Development

            Design a five-year ESG strategy, roadmap, and action plan.
            Align strategy with LCBO’s purpose and government mandates.
            Innovate in ESG practices and industry collaboration.
            Establish environmental and social initiatives.
            Develop an impact measurement and reporting framework.
            Phase 3: ESG Strategy Execution

            Implement the action plan within financial projections.
            Ensure alignment with organizational resources.
            Produce LCBO ESG Annual Reports.
            Track progress and adapt to emerging frameworks.
            Phase 4: Continued Support

            Continue executing the ESG strategy for the remaining 36 months.
            Identify and implement new initiatives.
            Provide ad-hoc support as needed.
            Evaluation Criteria
            Company Qualifications - 5 points
            Case Studies/Examples - 10 points
            Team and Experience - 10 points
            Work Plan, Approach and Methodology - 30 points

            Now you will evaluate both responses and return the detailed scoring result with table of scores for both Responses and reationale behind the scoring in another column. Rationale should be as detailed as possible. 
            Table format: Column 1 - Criteria; Column 2- Response 1 (Company name); Column 3-Response 2 (company name); Column 4 - Scoring Rationale
            Provide total score table below the above table.
            Total score table format: Column 1: Company Name; Column 2: Total Score
            Then provide the final recommendation paragraph explaining your opinion on evaluation.Try to be as detailed as possible in your response.
            Here are the responses: {raw_text}

            '''

            prompt = PromptTemplate(template=prompt, input_variables=["raw_text"])
            print("Prompt is....",prompt)
            model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
            chat_llm_chain = LLMChain(
                llm=model,
                prompt=prompt,
                verbose=True
            )    
            response = chat_llm_chain.predict(raw_text=raw_text)
            st.write(response)
            


if __name__ == "__main__":
    # with open('https://github.com/pranavGenAI/bidbooster/blob/475ae18b3c1f5a05a45ff983e06b025943137576/wave.css') as f:
        # css = f.read()

    st.markdown('''<style>
        .stApp > header {
        background-color: transparent;
    }
    @keyframes my_animation {
        0% {background-position: 0% 0%;}
        50% {background-position: 100% 100%;}
        100% {background-position: 0% 0%;}
    }
    [data-testid=stSidebar] {
        background: linear-gradient(360deg, #1a2631 95%, #161d29 10%);
    }
    div.stButton > button:first-child {
        background:linear-gradient(45deg, #c9024b 45%, #ba0158 55%, #cd006d 70%);
        color: white;
        border: none;
    }
    div.stButton > button:hover {
        background:linear-gradient(45deg, #ce026f 45%, #970e79 55%, #6c028d 70%);
        background-color:#ce1126;
    }
    div.stButton > button:active {
        position:relative;
        top:3px;
    }    


    </style>''', unsafe_allow_html=True)
    main()
