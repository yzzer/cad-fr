from utils.search import FaceSearchService

if __name__ == '__main__':
    service = FaceSearchService("./config/ds_model.pkl")
    print(len(service.find("./config/2014.jpg", threshold=0.35)))