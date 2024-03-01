#/bin/bash

# Append all the arguments to the command
python bin/llm_predict.py \
    --model_name "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ" $@ 
