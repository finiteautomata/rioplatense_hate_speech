from rioplatense_hs.config import config
from openai import OpenAI

API_KEY = config["OPENAI"]["API_KEY"]
client = OpenAI(api_key=API_KEY)


def get_completion(prompt, model="gpt-3.5-turbo"):
    return client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model=model
    )
