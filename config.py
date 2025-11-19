import streamlit as st

# Load WatsonX configuration from Streamlit secrets
WATSONX_API_KEY = st.secrets["watsonx"]["api_key"]
WATSONX_URL = st.secrets["watsonx"]["url"]
WATSONX_PROJECT_ID = st.secrets["watsonx"]["project_id"]

# Model configurations
COORDINATOR_ID = st.secrets["models"]["coordinator_id"]
COORDINATOR_PARAMETERS = {
    "decoding_method": st.secrets["models"]["coordinator_parameters"]["decoding_method"],
    "max_new_tokens": st.secrets["models"]["coordinator_parameters"]["max_new_tokens"],
    "temperature": st.secrets["models"]["coordinator_parameters"]["temperature"],
    "top_p": st.secrets["models"]["coordinator_parameters"]["top_p"],
    "top_k": st.secrets["models"]["coordinator_parameters"]["top_k"]
}

PREDICTIVE_ID = st.secrets["models"]["predictive_id"]
PREDICTIVE_PARAMETERS = {
    "decoding_method": st.secrets["models"]["predictive_parameters"]["decoding_method"],
    "max_new_tokens": st.secrets["models"]["predictive_parameters"]["max_new_tokens"],
    "temperature": st.secrets["models"]["predictive_parameters"]["temperature"],
    "repetition_penalty": st.secrets["models"]["predictive_parameters"]["repetition_penalty"]
}

SENTIMENT_ID = st.secrets["models"]["sentiment_id"]
SENTIMENT_PARAMETERS = {
    "decoding_method": st.secrets["models"]["sentiment_parameters"]["decoding_method"],
    "max_new_tokens": st.secrets["models"]["sentiment_parameters"]["max_new_tokens"],
    "min_new_tokens": st.secrets["models"]["sentiment_parameters"]["min_new_tokens"],
    "temperature": st.secrets["models"]["sentiment_parameters"]["temperature"],
    "top_k": st.secrets["models"]["sentiment_parameters"]["top_k"],
    "top_p": st.secrets["models"]["sentiment_parameters"]["top_p"],
    "repetition_penalty": st.secrets["models"]["sentiment_parameters"]["repetition_penalty"]
}