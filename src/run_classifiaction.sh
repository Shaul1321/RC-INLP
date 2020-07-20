layers='0 3 6 6-random0 6-random1 6-random2 6-random3 6-random4 9 12'
#layers='6-random2 6-random3 6-random4 9 12'
recall=1

for layer in $layers;do 
    echo "$layer"
    python3 train_classifiers.py --train-dev-path ../data/datasets.5000a.layer="$layer".masked=True.pickle --test-path ../data/datasets.5000t.layer="$layer".masked=True.pickle --recall "$recall" --classifier "sgd-log"
    echo "======================================="
done


