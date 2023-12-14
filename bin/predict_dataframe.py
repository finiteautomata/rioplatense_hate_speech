from tqdm.auto import tqdm
import pandas as pd
import fire
import time
from rioplatense_hs.prompting import get_response, build_base_prompt
from rioplatense_hs.preprocessing import text_to_label, labels


def predict_row(context, text, base_prompt, model_name="gpt-3.5-turbo"):
    try:
        return get_response(
            context,
            text,
            model=model_name,
            base_prompt=base_prompt,
        )
    # If rate limit is reached, wait 5 seconds and retry
    except Exception as e:
        print(f"Error: {e} -- {type(e)}")
        time.sleep(2)
        return None


def predict(df, model_name, base_prompt, max_retries=5):
    # Generate tasks

    ids = df.index.tolist()

    outs = {idx: None for idx in ids}

    retry_num = 1

    while any(o is None for o in outs.values()) and retry_num <= max_retries:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            if outs[idx] is not None:
                continue
            outs[idx] = predict_row(
                row["context_tweet"],
                row["text"],
                model_name=model_name,
                base_prompt=base_prompt,
            )

        # Get tqdm ordered results

        retry_num += 1

    return outs


def predict_dataframe(
    input,
    output,
    model_name="gpt-3.5-turbo",
    num_examples=None,
    shuffle=False,
    limit=None,
):
    print(
        f"Predicting {input} -- with num_examples={num_examples} and shuffle={shuffle}"
    )

    # Build base prompt

    base_prompt = build_base_prompt(num_examples=num_examples, shuffle=shuffle)

    print("=" * 80)
    print(f"Example prompt:\n{base_prompt}")
    df = pd.read_csv(input)

    if limit is not None:
        df = df.iloc[:limit]
    assert "context_tweet" in df.columns and "text" in df.columns

    # Start asyncio with predict

    outs = predict(df, model_name=model_name, base_prompt=base_prompt)

    df["prompt"] = df.index.map(lambda x: outs[x][0])
    df["pred_cot"] = df.index.map(lambda x: outs[x][1])

    pred_labels = [f"PRED_{l}" for l in labels]

    for idx, value in df["pred_cot"].items():
        preds = text_to_label(value)

        for k, v in preds.items():
            df.loc[idx, f"PRED_{k}"] = int(v)

    # Convert pred_labels to int
    for l in pred_labels:
        df[l] = df[l].astype(int)

    print(f"Saving to {output}")

    pred_hate = df[pred_labels].sum(axis=1) > 0

    df["PRED_HATEFUL"] = pred_hate.astype(int)

    df.to_csv(output, index=False)


if __name__ == "__main__":
    fire.Fire(predict_dataframe)
