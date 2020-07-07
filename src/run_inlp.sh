# all, trained bert, masked

python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=0.masked=True.pickle --num-classifiers 16 --classifier sgd --all 1
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=3.masked=True.pickle --num-classifiers 16 --classifier sgd --all 1
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=6.masked=True.pickle --num-classifiers 16 --classifier sgd --all 1
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=9.masked=True.pickle --num-classifiers 16 --classifier sgd --all 1
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=12.masked=True.pickle --num-classifiers 16 --classifier sgd --all 1

# all, random bert, masked

python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=6-random0.masked=True.pickle --num-classifiers 16 --classifier sgd --all 1
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=6-random1.masked=True.pickle --num-classifiers 16 --classifier sgd --all 1
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=6-random2.masked=True.pickle --num-classifiers 16 --classifier sgd --all 1
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=9-random3.masked=True.pickle --num-classifiers 16 --classifier sgd --all 1
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=6-random4.masked=True.pickle --num-classifiers 16 --classifier sgd --all 1

# trained bert, masked

python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=0.masked=True.pickle --num-classifiers 16 --classifier sgd
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=3.masked=True.pickle --num-classifiers 16 --classifier sgd
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=6.masked=True.pickle --num-classifiers 16 --classifier sgd
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=9.masked=True.pickle --num-classifiers 16 --classifier sgd
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=12.masked=True.pickle --num-classifiers 16 --classifier sgd

echo "finished trained bert, masked"

# untrained bert, masked

python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=6-random0.masked=True.pickle --num-classifiers 16 --classifier sgd
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=6-random1.masked=True.pickle --num-classifiers 16 --classifier sgd
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=6-random2.masked=True.pickle --num-classifiers 16 --classifier sgd
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=6-random3.masked=True.pickle --num-classifiers 16 --classifier sgd
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=6-random4.masked=True.pickle --num-classifiers 16 --classifier sgd

echo "finished untrained trained bert, masked"

# trained bert, unmasked

python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=0.masked=False.pickle --num-classifiers 16 --classifier sgd
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=3.masked=False.pickle --num-classifiers 16 --classifier sgd
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=6.masked=False.pickle --num-classifiers 16 --classifier sgd
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=9.masked=False.pickle --num-classifiers 16 --classifier sgd
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=12.masked=False.pickle --num-classifiers 16 --classifier sgd

echo "finished trained bert, unmasked"

# untrained bert, unmasked

python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=6-random0.masked=False.pickle --num-classifiers 16 --classifier sgd
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=6-random1.masked=False.pickle --num-classifiers 16 --classifier sgd
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=6-random2.masked=False.pickle --num-classifiers 16 --classifier sgd
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=6-random3.masked=False.pickle --num-classifiers 16 --classifier sgd
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=6-random4.masked=False.pickle --num-classifiers 16 --classifier sgd
