# trained bert, masked

python3 generate_datasets.py --sentences-group adapt --input-path ../data/data_with_states.layer=0.masked=True.model=bert.pickle
python3 generate_datasets.py --sentences-group test --input-path ../data/data_with_states.layer=0.masked=True.model=bert.pickle

python3 generate_datasets.py --sentences-group adapt --input-path ../data/data_with_states.layer=3.masked=True.model=bert.pickle
python3 generate_datasets.py --sentences-group test --input-path ../data/data_with_states.layer=3.masked=True.model=bert.pickle

python3 generate_datasets.py --sentences-group adapt --input-path ../data/data_with_states.layer=6.masked=True.model=bert.pickle
python3 generate_datasets.py --sentences-group test --input-path ../data/data_with_states.layer=6.masked=True.model=bert.pickle

python3 generate_datasets.py --sentences-group adapt --input-path ../data/data_with_states.layer=9.masked=True.model=bert.pickle
python3 generate_datasets.py --sentences-group test --input-path ../data/data_with_states.layer=9.masked=True.model=bert.pickle

python3 generate_datasets.py --sentences-group adapt --input-path ../data/data_with_states.layer=12.masked=True.model=bert.pickle
python3 generate_datasets.py --sentences-group test --input-path ../data/data_with_states.layer=12.masked=True.model=bert.pickle

# untrained bert, masked

python3 generate_datasets.py --sentences-group adapt --input-path ../data/data_with_states.layer=6-random0.masked=True.model=bert.pickle
python3 generate_datasets.py --sentences-group test --input-path ../data/data_with_states.layer=6-random0.masked=True.model=bert.pickle

python3 generate_datasets.py --sentences-group adapt --input-path ../data/data_with_states.layer=6-random1.masked=True.model=bert.pickle
python3 generate_datasets.py --sentences-group test --input-path ../data/data_with_states.layer=6-random1.masked=True.model=bert.pickle

python3 generate_datasets.py --sentences-group adapt --input-path ../data/data_with_states.layer=6-random2.masked=True.model=bert.pickle
python3 generate_datasets.py --sentences-group test --input-path ../data/data_with_states.layer=6-random2.masked=True.model=bert.pickle

python3 generate_datasets.py --sentences-group adapt --input-path ../data/data_with_states.layer=6-random3.masked=True.model=bert.pickle
python3 generate_datasets.py --sentences-group test --input-path ../data/data_with_states.layer=6-random3.masked=True.model=bert.pickle

python3 generate_datasets.py --sentences-group adapt --input-path ../data/data_with_states.layer=6-random4.masked=True.model=bert.pickle
python3 generate_datasets.py --sentences-group test --input-path ../data/data_with_states.layer=6-random4.masked=True.model=bert.pickle

# trained bert, unmasked

python3 generate_datasets.py --sentences-group adapt --input-path ../data/data_with_states.layer=0.masked=False.model=bert.pickle
python3 generate_datasets.py --sentences-group test --input-path ../data/data_with_states.layer=0.masked=False.model=bert.pickle

python3 generate_datasets.py --sentences-group adapt --input-path ../data/data_with_states.layer=3.masked=False.model=bert.pickle
python3 generate_datasets.py --sentences-group test --input-path ../data/data_with_states.layer=3.masked=False.model=bert.pickle

python3 generate_datasets.py --sentences-group adapt --input-path ../data/data_with_states.layer=6.masked=False.model=bert.pickle
python3 generate_datasets.py --sentences-group test --input-path ../data/data_with_states.layer=6.masked=False.model=bert.pickle

python3 generate_datasets.py --sentences-group adapt --input-path ../data/data_with_states.layer=9.masked=False.model=bert.pickle
python3 generate_datasets.py --sentences-group test --input-path ../data/data_with_states.layer=9.masked=False.model=bert.pickle

python3 generate_datasets.py --sentences-group adapt --input-path ../data/data_with_states.layer=12.masked=False.model=bert.pickle
python3 generate_datasets.py --sentences-group test --input-path ../data/data_with_states.layer=12.masked=False.model=bert.pickle


# untrained bert, unmasked

python3 generate_datasets.py --sentences-group adapt --input-path ../data/data_with_states.layer=6-random0.masked=False.model=bert.pickle
python3 generate_datasets.py --sentences-group test --input-path ../data/data_with_states.layer=6-random0.masked=False.model=bert.pickle

python3 generate_datasets.py --sentences-group adapt --input-path ../data/data_with_states.layer=6-random1.masked=False.model=bert.pickle
python3 generate_datasets.py --sentences-group test --input-path ../data/data_with_states.layer=6-random1.masked=False.model=bert.pickle

python3 generate_datasets.py --sentences-group adapt --input-path ../data/data_with_states.layer=6-random2.masked=False.model=bert.pickle
python3 generate_datasets.py --sentences-group test --input-path ../data/data_with_states.layer=6-random2.masked=False.model=bert.pickle

python3 generate_datasets.py --sentences-group adapt --input-path ../data/data_with_states.layer=6-random3.masked=False.model=bert.pickle
python3 generate_datasets.py --sentences-group test --input-path ../data/data_with_states.layer=6-random3.masked=False.model=bert.pickle

python3 generate_datasets.py --sentences-group adapt --input-path ../data/data_with_states.layer=6-random4.masked=False.model=bert.pickle
python3 generate_datasets.py --sentences-group test --input-path ../data/data_with_states.layer=6-random4.masked=False.model=bert.pickle
