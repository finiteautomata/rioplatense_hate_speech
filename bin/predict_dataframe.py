from tqdm.auto import tqdm
import asyncio
import pandas as pd
import fire
import time
from rioplatense_hs.prompting import build_prompt
from rioplatense_hs.openai import async_get_completion


async def predict_row(idx, context, text, model_name="gpt-3.5-turbo"):
    try:
        prompt = build_prompt(context, text)
        response = await async_get_completion(prompt, model=model_name)

        text = response.choices[0].message.content
        return (idx, prompt, text)
    # If rate limit is reached, wait 5 seconds and retry
    except Exception as e:
        print(f"Error: {e} -- {type(e)}")
        time.sleep(2)
        return (idx, prompt, None)


async def predict(df, model_name, max_retries=5):
    # Generate tasks

    ids = df.index.tolist()

    outs = {idx: None for idx in ids}

    retry_num = 1

    while any(o is None for o in outs.values()) and retry_num <= max_retries:
        tasks = [
            predict_row(idx, row["context_tweet"], row["text"], model_name=model_name)
            for idx, row in df.iterrows()
            if outs[idx] is None
        ]

        # Get tqdm ordered results

        pbar = tqdm(total=len(tasks))
        for f in asyncio.as_completed(tasks):
            idx, prompt, response = await f
            if response is not None:
                outs[idx] = (prompt, response)
            pbar.update(1)

        retry_num += 1

    return outs


def predict_dataframe(input, output, model_name="gpt-3.5-turbo"):
    print(f"Predicting {input}")

    df = pd.read_csv(input)

    assert "context_tweet" in df.columns and "text" in df.columns

    # Start asyncio with predict

    outs = asyncio.run(predict(df, model_name=model_name))

    df["prompt"] = df.index.map(lambda x: outs[x][0])
    df["pred_cot"] = df.index.map(lambda x: outs[x][1])

    print(f"Saving to {output}")
    df.to_csv(output, index=False)


if __name__ == "__main__":
    fire.Fire(predict_dataframe)
