import os
from cleanlab_tlm import TLM
from sklearn.metrics import accuracy_score
import requests
from dotenv import load_dotenv
load_dotenv()
CLEANLAB_TLM_API_KEY = os.getenv("CLEANLAB_TLM_API_KEY")
tlm = TLM(options={"log": ["explanation"], "model": "gpt-4.1-mini"}) 

question = "how many r's are there in 'strawberry'"
output = tlm.prompt(question)

print(f'Response: {output["response"]}')
print(f'Trustworthiness Score: {output["trustworthiness_score"]}')
print(f'Explanation: {output["log"]["explanation"]}')
