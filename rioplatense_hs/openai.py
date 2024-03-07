from rioplatense_hs.config import config
from openai import OpenAI, AsyncOpenAI, RateLimitError, APITimeoutError
import os
import time


# I get the API_KEY from the environment variable OPENAI_API_KEY, if it's not there, I get it from the config file
API_KEY = os.environ.get("OPENAI_API_KEY", config["OPENAI"]["API_KEY"])
client = OpenAI(api_key=API_KEY, max_retries=3)
async_client = AsyncOpenAI(api_key=API_KEY, max_retries=10)


def get_completion(prompt, model="gpt-3.5-turbo-0125"):
    return client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model=model
    )


async def async_get_completion(prompt, model="gpt-3.5-turbo", max_retries=10):
    retry_num = 0
    wait_time = 2
    while retry_num <= max_retries:
        try:
            return await async_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}], model=model
            )
        except RateLimitError as e:
            print(f"Error: {e} -- type: {type(e)}")
            time.sleep(wait_time)
            wait_time *= 2
            retry_num += 1
        except APITimeoutError as e:
            print(f"Error: {e} -- type: {type(e)}")
            time.sleep(wait_time)
            wait_time *= 2
            retry_num += 1
