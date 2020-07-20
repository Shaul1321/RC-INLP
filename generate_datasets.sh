# trained bert, masked

python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=0.masked=True.pickle
python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=0.masked=True.pickle

python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=3.masked=True.pickle
python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=3.masked=True.pickle

python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=6.masked=True.pickle
python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=6.masked=True.pickle

python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=9.masked=True.pickle
python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=9.masked=True.pickle

python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=12.masked=True.pickle
python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=12.masked=True.pickle

# untrained bert, masked

python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=6-random0.masked=True.pickle
python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=6-random0.masked=True.pickle

python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=6-random1.masked=True.pickle
python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=6-random1.masked=True.pickle

python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=6-random2.masked=True.pickle
python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=6-random2.masked=True.pickle

python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=6-random3.masked=True.pickle
python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=6-random3.masked=True.pickle

python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=6-random4.masked=True.pickle
python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=6-random4.masked=True.pickle

# trained bert, unmasked

python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=0.masked=False.pickle
python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=0.masked=False.pickle

python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=3.masked=False.pickle
python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=3.masked=False.pickle

python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=6.masked=False.pickle
python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=6.masked=False.pickle

python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=9.masked=False.pickle
python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=9.masked=False.pickle

python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=12.masked=False.pickle
python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=12.masked=False.pickle


# untrained bert, unmasked

python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=6-random0.masked=False.pickle
python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=6-random0.masked=False.pickle

python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=6-random1.masked=False.pickle
python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=6-random1.masked=False.pickle

python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=6-random2.masked=False.pickle
python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=6-random2.masked=False.pickle

python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=6-random3.masked=False.pickle
python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=6-random3.masked=False.pickle

python3 generate_datasets.py --sentences-group 5000a --input-path ../data/data_with_states.layer=6-random4.masked=False.pickle
python3 generate_datasets.py --sentences-group 5000t --input-path ../data/data_with_states.layer=6-random4.masked=False.pickle
