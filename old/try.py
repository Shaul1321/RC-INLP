import pickle

with open("../data/data.pickle", "rb") as f:
    data = pickle.load(f)

q = [d for d in data if d["label"] == 0 and d["type_neg"] == "outside-rc"]
q2 = [d for d in data if d["label"] == 0 and d["type_neg"] == "non-rc-random"]
q3 = [d for d in data if d["label"] == 0 and d["type_neg"] == "non-rc-corresponding"]
Q = [d for d in data if d["label"] == 0]

print(len(q), len(q2), len(q3), len(Q)/len(data) )

for i in range(25):

    print(data[i])
    print("==================================================")
