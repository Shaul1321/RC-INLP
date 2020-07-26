# trained bert, masked

python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=0.masked=True.model=bert.pickle --num-classifiers 10 --classifier sgd-log
python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=3.masked=True.model=bert.pickle --num-classifiers 10 --classifier sgd-log
#python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=4.masked=True.model=bert.pickle --num-classifiers 10 --classifier sgd-log
#python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=5.masked=True.model=bert.pickle --num-classifiers 10 --classifier sgd-log
python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=6.masked=True.model=bert.pickle --num-classifiers 10 --classifier sgd-log
python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=9.masked=True.model=bert.pickle --num-classifiers 10 --classifier sgd-log
python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=12.masked=True.model=bert.pickle --num-classifiers 10 --classifier sgd-log

echo "finished trained bert, masked"

# untrained bert, masked

python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=6-random0.masked=True.model=bert.pickle --num-classifiers 10 --classifier sgd-log
python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=6-random1.masked=True.model=bert.pickle --num-classifiers 10 --classifier sgd-log
python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=6-random2.masked=True.model=bert.pickle --num-classifiers 10 --classifier sgd-log
python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=6-random3.masked=True.model=bert.pickle --num-classifiers 10 --classifier sgd-log
python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=6-random4.masked=True.model=bert.pickle --num-classifiers 10 --classifier sgd-log

echo "finished untrained bert, masked"

exit 0

# trained bert, unmasked

python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=0.masked=False.model=bert.pickle --num-classifiers 10 --classifier svm
python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=3.masked=False.model=bert.pickle --num-classifiers 10 --classifier svm
python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=6.masked=False.model=bert.pickle --num-classifiers 10 --classifier svm
python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=9.masked=False.model=bert.pickle --num-classifiers 10 --classifier svm
python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=12.masked=False.model=bert.pickle --num-classifiers 10 --classifier svm

echo "finished trained bert, unmasked"

# untrained bert, unmasked

python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=6-random0.masked=False.model=bert.pickle --num-classifiers 10 --classifier svm
python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=6-random1.masked=False.model=bert.pickle --num-classifiers 10 --classifier svm
python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=6-random2.masked=False.model=bert.pickle --num-classifiers 10 --classifier svm
python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=6-random3.masked=False.model=bert.pickle --num-classifiers 10 --classifier svm
python3 run_inlp.py --train-dev-path ../data/datasets.adapt.layer=6-random4.masked=False.model=bert.pickle --num-classifiers 10 --classifier svm
