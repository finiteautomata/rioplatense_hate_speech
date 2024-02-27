import fire
from tqdm.auto import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pysentimiento.preprocessing import preprocess_tweet as pysent_preprocess
from rioplatense_hs.preprocessing import preprocess_tweet
from datasets import Dataset

template_prompt = """[INST] Determinar si el siguiente mensaje contiene discurso de odio.

Entendemos que hay discurso de odio en el tweet si tiene declaraciones de carácter intenso e irracional de rechazo, enemistad y aborrecimiento contra un individuo o contra un grupo, siendo objetivos de estas expresiones por poseer una característica protegida. Las características protegidas que contemplamos son:

- mujer: mujeres o movimiento feminista
- lgbti: contra gays, lesbianas, transexuales y otras identidades de género
- racismo: inmigrantes, xenofobia, o contra pueblos aborígenes
- clase: personas de bajos recursos o motivos de clase
- discapacidad: discapacidad, problemas de salud mental o de adicción al alcohol u otras drogas
- política: ideología política
- aspecto: aspecto, gordofobia o edad
- criminal: presos o delincuentes comunes

Responder una o varias de las características separadas por coma, o bien "nada" si no hay discurso de odio. Pensar paso a paso la respuesta antes de responder.
[/INST]
contexto: Al borde del llanto, Nati Jota pidió disculpas por sus repudiables tuits pero los justificó: 'Quería sumar seguidores'
texto: Show o nada mi amorrr!! Sos una imbesil!
salida: Es un insulto y no menciona nada relacionado a las características protegidas. La respuesta final es 'nada'
---
contexto: Violento motín en la cárcel de Devoto: tomaron e incendiaron pabellones
texto: Hermanito lo que son esos grones. Ahí adentro le robaron la billetera al coronavirus
salida: La palabra "grones" significa negros, lo cual alude a su color de piel y aspecto. Además, se refiere a su condición de presos. La respuesta final es "criminal, racismo, aspecto"
---
contexto: Mayra Mendoza criticó el banderazo contra el Gobierno: "No tiene sentido y es antidemocrático"
texto: Que negrita villera....ni con la plata q roba puede arreglarse
salida: El texto alude a su color de piel, a su condición social (villera), y a su aspecto. La respuesta final es "racismo, aspecto, clase"
---
contexto: Les darán DNI provisorio a personas en situación vulnerable, para que puedan empezar a cobrar planes sociales
texto: Seguimos alimentando vagos
salida: El comentario se refiere a quienes cobran planes sociales como vagos. La respuesta final es "clase"
---
contexto: Otro ataque de furia contra un colectivero: una mujer trans le pateó la unidad y le tiró un ladrillazo
texto: Un tipo operado. Con la fuerza de un hombre y no la de una mujer
salida: El texto alude a que la mujer trans es un hombre. La respuesta final es "lgbti"
---
contexto: Elisa Carrió denunció que el Gobierno usa la pandemia para "establecer un estado de sitio"
texto: Gorda psiquiátrica
salida: El texto alude a su aspecto (gorda) y la acusa de tener problemas psiquiátricos. La respuesta final es "aspecto, discapacidad"
---
contexto: Los dos presos heridos de bala en el motín de Devoto tienen Covid-19 y uno quedó hemipléjico
texto: justicia divina!
salida: El texto alude a que los presos merecen ser baleados. La respuesta final es 'criminal'
</s>
[INST]
contexto: {contexto}
texto: {texto}
[/INST]
"""

def llm_predict(
    input,
    output_path,
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    batch_size=8,
):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )
    df = pd.read_csv(input, index_col=0)
    tokenizer.model_max_length = 6400
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Iterate over the batches and predict

    test_ds = Dataset.from_pandas(df)
    #
    def tokenize(example):
        texto = preprocess_tweet(example["text"])

        contexto = pysent_preprocess(
            example["context_tweet"],
            preprocess_hashtags=False,
            demoji=False,
            preprocess_handles=False,
        )
        model_input = template_prompt.format(
            contexto=contexto, texto=texto
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


    dataloader = DataLoader(
        sorted_tokenized_ds,
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=False,
        pin_memory=True,
        num_workers=16,
    )

    outs = {}
    for ids, inputs in tqdm(dataloader):
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        output = model.generate(
            **inputs,
            max_new_tokens=150, pad_token_id=tokenizer.eos_token_id
        )

        for k, id in enumerate(ids):
            output_text = tokenizer.decode(output[k], skip_special_tokens=True)

            outs[id] = output_text

    assert len(outs) == len(df)

    df["prompt"] = df.index.map(outs)

    df["salida"] = df["prompt"].apply(
        lambda x: x.split("[/INST]")[-1].strip().replace("\n", ""))
    df.to_csv(output_path)

    print(f"Predictions saved to {output_path} -- len {len(df)}")



if __name__ == "__main__":
    fire.Fire(llm_predict)
