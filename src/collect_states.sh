# trained bert, masked

python3 collect_bert_states.py --layer 0 --mask 1 --random 0
python3 collect_bert_states.py --layer 3 --mask 1 --random 0
python3 collect_bert_states.py --layer 6 --mask 1 --random 0
python3 collect_bert_states.py --layer 9 --mask 1 --random 0
python3 collect_bert_states.py --layer 12 --mask 1 --random 0

echo "finished trained bert, masked"

# untrained bert, masked

python3 collect_bert_states.py --layer 6 --mask 1 --random 1 --label 0
python3 collect_bert_states.py --layer 6 --mask 1 --random 1 --label 1
python3 collect_bert_states.py --layer 6 --mask 1 --random 1 --label 2
python3 collect_bert_states.py --layer 6 --mask 1 --random 1 --label 3
python3 collect_bert_states.py --layer 6 --mask 1 --random 1 --label 4

echo "finished untrained trained bert, masked"

# trained bert, unmasked

python3 collect_bert_states.py --layer 0 --mask 0 --random 0
python3 collect_bert_states.py --layer 3 --mask 0 --random 0
python3 collect_bert_states.py --layer 6 --mask 0 --random 0
python3 collect_bert_states.py --layer 9 --mask 0 --random 0
python3 collect_bert_states.py --layer 12 --mask 0 --random 0

echo "finished trained bert, unmasked"

# untrained bert, unmasked

python3 collect_bert_states.py --layer 6 --mask 0 --random 1 --label 0
python3 collect_bert_states.py --layer 6 --mask 0 --random 1 --label 1
python3 collect_bert_states.py --layer 6 --mask 0 --random 1 --label 2
python3 collect_bert_states.py --layer 6 --mask 0 --random 1 --label 3
python3 collect_bert_states.py --layer 6 --mask 0 --random 1 --label 4

echo "finished untrained bert, unmasked."
