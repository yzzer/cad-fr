import pickle

data = pickle.load(open('config/ds_model_VGG-Face_75.pkl', 'rb'))
print(len(data))  # 3563