# alpha=5, layer=6


python3 test_on_agreement.py --only-attractors 1 --layer 6 --alpha -4 --iters 8 --classifier sgd-log
python3 test_on_agreement.py --only-attractors 1 --layer 6 --alpha 4 --iters 8 --classifier sgd-log
python3 test_on_agreement.py --only-not-attractors 1 --layer 6 --alpha 4 --iters 8 --classifier sgd-log
python3 test_on_agreement.py --only-not-attractors 1 --layer 6 --alpha -4 --iters 8 --classifier sgd-log

# alpha = 0, layer 6

python3 test_on_agreement.py --only-attractors 1 --layer 6 --alpha 0 --iters 8 --classifier sgd-log
python3 test_on_agreement.py --only-not-attractors 1 --layer 6 --alpha 0 --iters 8 --classifier sgd-log
