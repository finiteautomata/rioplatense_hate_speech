#/bin/bash

# Append all the arguments to the command

python bin/llm_predict.py \
    --model_name "CohereForAI/aya-101" --num_examples 7 --fp16 --seq2seq $@
