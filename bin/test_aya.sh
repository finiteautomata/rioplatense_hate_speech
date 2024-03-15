#/bin/bash
# Predict for all test_0i.csv files

for i in {1..5}
do
    path="data/test_0${i}.csv"
    out="data/runs/test_0${i}_aya.csv"
    echo "Predicting for test_0$i.csv -- output: $out"
    bash bin/predict_aya.sh --input $path --output_path $out --batch_size 8
done

