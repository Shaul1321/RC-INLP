layers='0 3 6 6-random0 6-random1 6-random2 6-random3 6-random4 9 12'
layers='6'
recall=0
model='roberta'

for layer in $layers;do 
    echo "$layer"
    python3 train_classifiers.py --train-dev-path ../data/datasets.adapt.layer="$layer".masked=True.model="$model".pickle --test-path ../data/datasets.test.layer="$layer".masked=True.model="$model".pickle --recall "$recall" --classifier "sgd-log"
    echo "======================================="
done

