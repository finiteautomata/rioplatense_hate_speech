#/bin/bash

# Append all the arguments to the command
# if --quantize, set model to TheBloke
if [[ $@ == *"--gptq8"* ]]; then
    model_name="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"

    revision="gptq-8bit-128g-actorder_True"
    # Remove the --gptq8 flag
    set -- "${@/--gptq8/}"
    # Add the revision
    set -- "${@} --revision ${revision}"
elif [[ $@ == *"--gptq4"* ]]; then
    model_name="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"
    revision="gptq-4bit-32g-actorder_True"
    # Remove the --gptq4 flag
    set -- "${@/--gptq4/}"
    # Add the revision
    set -- "${@} --revision ${revision}"
elif [[ $@ == *"--gptq"* ]]; then
    model_name="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"
    # Remove the --gptq flag
    set -- "${@/--gptq/}"
else
    # Set mixtral
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"
fi

# If batch size is not set, set it to 2
if [[ $@ != *"--batch_size"* ]]; then
    set -- "${@} --batch_size 2"
fi

python bin/llm_predict.py \
    --model_name $model_name $@
