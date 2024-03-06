#/bin/bash
# Predict for all test_0i.csv files

for i in {1..5}
do
    path="data/test_0${i}.csv"
    out="data/test_0${i}_mixtral.csv"
    echo "Predicting for test_0$i.csv -- output: $out"
    bash bin/predict_amd.sh --gptq4 --input $path --output_path $out --batch_size 8
done

