#sh run_inlp.sh
#python3 eval_rowspace.py --iters 10 --classifier sgd-log --random 1
python3 test_on_agreement.py --only-attractors 1 --layer 6-random0 --alpha 4 --iter 10 --classifier sgd-log
python3 test_on_agreement.py --only-attractors 1 --layer 6-random0 --alpha -4 --iter 10 --classifier sgd-log

python3 test_on_agreement.py --only-not-attractors 1 --layer 6-random0 --alpha 4 --iter 10 --classifier sgd-log
python3 test_on_agreement.py --only-not-attractors 1 --layer 6-random0 --alpha -4 --iter 10 --classifier sgd-log

python3 test_on_agreement.py --only-attractors 1 --layer 6-random0 --alpha 0 --iter 10 --classifier sgd-log
python3 test_on_agreement.py --only-not-attractors 1 --layer 6-random0 --alpha 0 --iter 10 --classifier sgd-log

#python3 test_on_agreement.py --only-attractors 1 --layer 6 --alpha 4 --iter 10 --classifier sgd-log
#python3 test_on_agreement.py --only-not-attractors 1 --layer 6 --alpha 4 --iter 10 --classifier sgd-log

#python3 test_on_agreement.py --only-attractors 1 --layer 6 --alpha -4 --iter 10 --classifier sgd-log
#python3 test_on_agreement.py --only-not-attractors 1 --layer 6 --alpha -4 --iter 10 --classifier sgd-log

#python3 test_on_agreement.py --only-attractors 1 --layer 6 --alpha 0 --iter 10 --classifier sgd-log
#python3 test_on_agreement.py --only-not-attractors 1 --layer 6 --alpha 0 --iter 10 --classifier sgd-log
