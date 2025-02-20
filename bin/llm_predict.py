import fire
import logging
import pandas as pd
import torch
import os
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from pysentimiento.preprocessing import preprocess_tweet as pysent_preprocess
from rioplatense_hs.preprocessing import preprocess_tweet, text_to_label, labels
from rioplatense_hs.mixtral import get_prompt as mixtral_get_prompt
from rioplatense_hs.tasks.hate_speech import build_prompt
from datasets import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set console output
# Add timestamp to logs
logger.addHandler(logging.StreamHandler())




def llm_predict(
    input,
    output,
    model_name,
    batch_size=8,
    max_new_tokens=512,
    do_sample=False,
    quantize=False,
    load_in_4bit=False,
    load_in_8bit=False,
    revision=None,
    top_p=0.95,
    seq2seq=False,
    num_examples=None,
    fp16=True,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    output_path = output
    model_args = {
        "device_map": "auto",
    }

    if quantize:
        model_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif fp16:
        model_args["torch_dtype"] = torch.float16

    logger.info(f"Predicting with model {model_name} -- revision {revision} -- num_examples {num_examples} -- batch_size {batch_size} -- max_new_tokens {max_new_tokens} -- do_sample {do_sample} -- top_p {top_p} -- fp16 {fp16} -- quantize {quantize} -- load_in_4bit {load_in_4bit} -- load_in_8bit {load_in_8bit}")
    if revision:
        logger.info(f"Using revision {revision}")
        model_args["revision"] = revision


    logger.info(f"Loading model...")

    auto_klass = AutoModelForSeq2SeqLM if seq2seq else AutoModelForCausalLM


    model = auto_klass.from_pretrained(
        model_name,
        **model_args,
    )

    logger.info(f"Model loaded")
    model.eval()

    if "gptq" in model_name.lower():
        try:
            from auto_gptq import exllama_set_max_input_length
            model = exllama_set_max_input_length(model, 4096 * 16)
        except:
            logger.info("Model does not support exllama_set_max_input_length")
    df = pd.read_csv(input, index_col=0)
    tokenizer.model_max_length = 6400
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Iterate over the batches and predict

    test_ds = Dataset.from_pandas(df)
    #

    if "mixtral" in model_name.lower():
        get_prompt = mixtral_get_prompt
    else:
        get_prompt = lambda context, text: build_prompt(
            contexto=context, texto=text, num_examples=num_examples
        )

    def tokenize(example):
        texto = preprocess_tweet(example["text"])

        contexto = pysent_preprocess(
            example["context_tweet"],
            preprocess_hashtags=False,
            demoji=False,
            preprocess_handles=False,
        )
        model_input = get_prompt(
            context=contexto, text=texto
        )
        return tokenizer(model_input, truncation=True)


    tokenized_ds = test_ds.map(tokenize, batched=False)
    # prompt: Sort tokenized_ds by length of input_ids
    tokenized_ds = tokenized_ds.map(lambda x: {"len": len(x["input_ids"])}, batched=False)

    sorted_tokenized_ds = tokenized_ds.sort("len")

    def collate(inputs):
        attention = [ex["attention_mask"] for ex in inputs]
        input_ids = [ex["input_ids"] for ex in inputs]
        ids = [ex["id"] for ex in inputs]
        return ids, tokenizer.pad({"input_ids": input_ids, "attention_mask": attention}, return_tensors="pt")

    # print max length

    logger.info(f"Max length: {max(tokenized_ds['len'])}")

    dataloader = DataLoader(
        sorted_tokenized_ds,
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=False,
        pin_memory=True,
        num_workers=16,
    )

    outs = {}
    with torch.no_grad():
        for ids, inputs in tqdm(dataloader):
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id,
                do_sample=do_sample, top_p=top_p
            )

            for k, id in enumerate(ids):
                output_text = tokenizer.decode(output[k], skip_special_tokens=True)
                logger.debug(output_text)

                outs[id] = output_text

    assert len(outs) == len(df)

    df["prompt"] = df.index.map(outs)

    df["pred_cot"] = df["prompt"].apply(
        lambda x: x.split("[/INST]")[-1].strip().replace("\n", ""))

    pred_labels = [f"PRED_{l}" for l in labels]

    for idx, value in df["pred_cot"].items():
        preds = text_to_label(value)

        for k, v in preds.items():
            df.loc[idx, f"PRED_{k}"] = int(v)


    # Convert pred_labels to int
    for l in pred_labels:
        df[l] = df[l].astype(int)

    pred_hate = df[pred_labels].sum(axis=1) > 0

    df["PRED_HATEFUL"] = pred_hate.astype(int)
    df["error"] = df["PRED_HATEFUL"] != df["HATEFUL"]

    print(f"Predictions saved to {output_path} -- len {len(df)}")
    df.to_csv(output_path)



if __name__ == "__main__":
    fire.Fire(llm_predict)
