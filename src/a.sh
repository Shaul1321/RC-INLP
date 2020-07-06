#python3 collect_bert_states.py --layer 6 --mask 0 --random 1
#python3 collect_bert_states.py --layer 6 --mask 1 --random 1
#python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=6-random.masked=True.pickle
#python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=6-random.masked=True.pickle
#python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=6-random.masked=False.pickle
#python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=6-random.masked=False.pickle

#python3 collect_bert_states.py --layer 3 --mask 0
#python3 collect_bert_states.py --layer 3 --mask 1
#python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=3.masked=True.pickle
#python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=3.masked=True.pickle
#python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=3.masked=False.pickle
#python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=3.masked=False.pickle


#python3 collect_bert_states.py --layer 6 --mask 0
#python3 collect_bert_states.py --layer 6 --mask 1
#python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=6.masked=True.pickle
#python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=6.masked=True.pickle
#python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=6.masked=False.pickle
#python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=6.masked=False.pickle
#python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=6-random.masked=False.pickle
#python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=0.masked=False.pickle
#python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=0.masked=False.pickle
#python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=3.masked=False.pickle
python3 collect_bert_states.py --layer 0 --mask 0 --random 1
python3 run_inlp.py --train-dev-path ../data/datasets.5000a.layer=0-random.masked=False.pickle
#python3 test_on_agreement.py
