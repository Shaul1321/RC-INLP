# alpha=5, layer=6

python3 test_on_agreement.py --only-attractors 1 --layer 3 --alpha 2 --iter 8 --classifier sgd-log
python3 test_on_agreement.py --only-attractors 1 --layer 3 --alpha -2 --iter 8 --classifier sgd-log
python3 test_on_agreement.py --only-not-attractors 1 --layer 3 --alpha 2 --iter 8 --classifier sgd-log
python3 test_on_agreement.py --only-not-attractors 1 --layer 3 --alpha -2 --iter 8 --classifier sgd-log

# alpha = 0, layer 6

python3 test_on_agreement.py --only-attractors 1 --layer 12 --alpha 0 --iter 8 --classifier sgd-log
python3 test_on_agreement.py --only-not-attractors 1 --layer 12 --alpha 0 --iter 8 --classifier sgd-log

exit 0

# alpha=2, layer=6

python3 test_on_agreement.py --only-attractors 1 --layer 6 --alpha 2.5 --iter 8 --classifier sgd-log
python3 test_on_agreement.py --only-attractors 1 --layer 6 --alpha -2.5 --iter 8 --classifier sgd-log
python3 test_on_agreement.py --only-not-attractors 1 --layer 6 --alpha 2.5 --iter 8 --classifier sgd-log
python3 test_on_agreement.py --only-not-attractors 1 --layer 6 --alpha -2.5 --iter 8 --classifier sgd-log


# alpha=2, layer=9

python3 test_on_agreement.py --only-attractors 1 --layer 9 --alpha 2.5 --iter 8 --classifier sgd-log
python3 test_on_agreement.py --only-attractors 1 --layer 9 --alpha -2.5 --iter 8 --classifier sgd-log
python3 test_on_agreement.py --only-not-attractors 1 --layer 9 --alpha 2.5 --iter 8 --classifier sgd-log
python3 test_on_agreement.py --only-not-attractors 1 --layer 9 --alpha -2.5 --iter 8 --classifier sgd-log

# alpha = 0, layer 6

python3 test_on_agreement.py --only-attractors 1 --layer 6 --alpha 0 --iter 8 --classifier sgd-log
python3 test_on_agreement.py --only-not-attractors 1 --layer 6 --alpha 0 --iter 8 --classifier sgd-log

# alpha=0, layer 9

python3 test_on_agreement.py --only-attractors 1 --layer 9 --alpha 0 --iter 8 --classifier sgd-log
python3 test_on_agreement.py --only-not-attractors 1 --layer 9 --alpha 0 --iter 8 --classifier sgd-log
