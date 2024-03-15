#/bin/bash

# Append all the arguments to the command

# Nota JM: pongo 7 ejemplos porque si no, por algún motivo no funciona
# i.e. devuelve "El texto" y la queda ahí

python bin/llm_predict.py \
    --model_name "CohereForAI/aya-101" --num_examples 7 --fp16 --seq2seq $@
