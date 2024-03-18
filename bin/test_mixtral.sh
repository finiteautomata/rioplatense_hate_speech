#/bin/bash
# Predict for all test_0i.csv files

for i in {1..5}
do
    path="data/test_0${i}.csv"
    out="data/runs/test_0${i}_mixtral.csv"
    echo "Predicting for test_0$i.csv -- output: $out"
    python bin/llm_predict.py --model_name mistralai/Mixtral-8x7B-Instruct-v0.1 --quantize --load_in_4bit --input $path --output $out --batch_size 8
done

