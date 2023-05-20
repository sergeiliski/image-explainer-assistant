# Import libraries & packages
import os 
import requests
from requests.exceptions import HTTPError
from urllib.parse import urlparse
import streamlit as st
from dotenv import load_dotenv


from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from langchain.utilities.scenexplain import SceneXplainAPIWrapper

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPEN_API_KEY")

scenex_api_key = os.getenv("SCENEX_API_KEY")
scenex_api_key_v2 = os.getenv("SCENEX_API_KEY_V2")
scenex_api_url: str = (
        "https://us-central1-causal-diffusion.cloudfunctions.net/describe"
    )

def get_image_name_from_url(url):
    path = urlparse(url).path
    return os.path.basename(path)

def describe_image(image: str) -> str:
    """Describe an image using the SceneXplain API."""
    try :
        return _describe_image(image, scenex_api_key)
    except HTTPError as error:
        if error.response.status_code == 400:
            print("API key is invalid, trying V2")
            try:
                return _describe_image(image, scenex_api_key_v2)
            except HTTPError as error:
                if error.response.status_code == 400:
                    print("V2 API key is invalid, try again later")
                    return ""
                else:
                    raise error

def _describe_image(image: str, key: str) -> str:
    headers = {
        "x-api-key": f"token {key}",
        "content-type": "application/json",
    }
    payload = {
        "data": [
            {
                "image": image,
                "algorithm": "Ember",
                "languages": ["en"],
            }
        ]
    }
    response = requests.post(scenex_api_url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json().get("result", [])
    img = result[0] if result else {}

    return img.get("text", "")

# Streamlet App framework
image_url = 'https://jina-ai-gmbh.ghost.io/content/images/2023/04/Jina-AI-Website-Banners-Templates---2023-04-19T150542.636-1.png'
st.title('Image explainer') # setting teh title
st.image(image_url) # set the featured image of the web application
prompt_url = st.text_input('Image URL') # The box for the image url
prompt = st.text_input('Ask your question')  # The box for the text prompt

question_template = PromptTemplate(
    input_variables = ['question', 'description'],
    template='you are an expert text analyzer. based on the following description: "{description}", answer the following question: "{question}". if i make a wrong statement, correct me. Do not mention about the description. Refer to the image instead.'
)

# Memory 
question_memory = ConversationBufferMemory(input_key='question', memory_key='chat_history')

# Llms
llm = OpenAI(temperature=0.9) 
question_chain = LLMChain(llm=llm, prompt=question_template, verbose=True, output_key='question', memory=question_memory)

# scene = SceneXplainAPIWrapper()
question = None
image_description_v2 = None

if  st.button('Submit') and prompt:
    print("Submitting...")
    image_description_v2 = describe_image(prompt_url)
    question = question_chain.run(question=prompt, description=image_description_v2)

if prompt_url:
    st.image(prompt_url, caption=get_image_name_from_url(prompt_url))

if question:
    st.markdown(question)

if image_description_v2:
    with st.expander('Image Description V2'):
        st.info(image_description_v2)