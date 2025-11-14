WATSONX_API_KEY = "AaTSagg_V5MFzhd7Obl1JWucKOs01zD_YkpokfKLFqvP"
WATSONX_URL = "https://us-south.ml.cloud.ibm.com/"
WATSONX_PROJECT_ID = "4d06cf15-5f29-4c58-8722-7c8a3caf383c"

COORDINATOR_ID = "ibm/granite-3-8b-instruct"
COORDINATOR_PARAMETERS = {
    "decoding_method": "greedy",
    "max_new_tokens": 2000,
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 50
}

PREDICTIVE_ID = "meta-llama/llama-3-3-70b-instruct"
PREDICTIVE_PARAMETERS = {
    "decoding_method": "greedy",
    "max_new_tokens": 1500,
    "temperature": 0.3,
    "repetition_penalty": 1.1
}

SENTIMENT_ID = "meta-llama/llama-3-3-70b-instruct"
SENTIMENT_PARAMETERS = {
    "decoding_method": "greedy",
    "max_new_tokens": 1500,
    "min_new_tokens": 1,
    "temperature": 0.3,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.2
}