import pickle

data = pickle.load(open('config/ds_model.pkl', 'rb'))
print(len(data))  # 3563