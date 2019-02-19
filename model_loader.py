import pickle
import os 

basedir = os.path.abspath(os.path.dirname(__file__))

if(os.name != 'posix'):
    SEA_MODEL_PATH=os.path.join(basedir, 'sources/sea_model_v1.pickle').replace('\\', '/')
    SEA_VOCABULARY_PATH =os.path.join(basedir, 'sources/sea_vocabulary_v1.pickle').replace('\\', '/')

else:
    SEA_MODEL_PATH =os.path.join(basedir, 'sources/sea_model_v1.pickle')
    SEA_VOCABULARY_PATH =os.path.join(basedir, 'sources/sea_vocabulary_v1.pickle')

class ModelLoader():
    """
    Loading trained model and vocabulary as instance which can be called later
    """
    def __init__(self):
        #Pickle model files for Sentiment classification

        # with open('sources/sea_vocabulary_v1.pickle', 'rb') as fp:
        #     self.sea_vocabulary= pickle.load(fp)

        # with open('sources/sea_model_v1.pickle', 'rb') as fp:
        #     self.sea_model_instance= pickle.load(fp)

        with open(SEA_VOCABULARY_PATH, 'rb') as fp:
            self.sea_vocabulary= pickle.load(fp)

        with open(SEA_MODEL_PATH, 'rb') as fp:
            self.sea_model_instance= pickle.load(fp)

    def get_sea_model(self):
        return self.sea_model_instance

    def get_sea_vocabulary(self):
        return self.sea_vocabulary