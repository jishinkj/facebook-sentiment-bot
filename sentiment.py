import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import RegexpTokenizer
from nltk.tokenize import word_tokenize
import re

class SeaModeling(object):
    def __init__(self, model, vocabulary):
        self.model_instance = model
        self.vocabulary = vocabulary
        self.builtinstopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                                 "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                                 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
                                 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
                                 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
                                 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
                                 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                                 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                                 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
                                 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                                 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                                 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
                                 'will', 'just', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'ma', 'ect',
                                 'enron', 'www', 'http', 'com', 'cc','image','pm','issues','work','would','hou','one']
                    # 'price','subject','day','meeting','energy','day','group','corp','service','company','week','year',
                    # 'date','us','error','occurred','transaction','call','business','deal',"change",'let','ga','font',
                    # 'houston','sans','serif',"problem",'mail','hotel', 'verdana','report','system','borland','initialize',
                    # 'name','time','development','market','employee','said','question','comment',"database",'engine','message',
                    # 'said','back','gas','email','going','send','verdana','trading', 'office','number','src','width','height',
                    # 'monday','friday','customer','review','deals','two','trading','getbad','transactions','end','sale',
                    # 'attempting','th','services','today','game','employees','ees','images','sportsline','thursday','comments',
                    # 'management','john','bpa','gov','jeff','key','fax','talk','eb','not','fyi','nan','get','kay','sara','north',
                    # 'vince','body','color','size','border','color','float','bgcolor','border','target','style','encoding',
                    # 'class','left','charset','center','type','en','padding','dear','hi','hey','hello']
        self.binaryReverseReplacement = {0: 'neutral', 1: "positive", 2: 'negative'}

    def preprocess(self, text):
        tokenized_text = word_tokenize(text)
        cleaned_text = " ".join([t for t in tokenized_text if re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)])
        cleaned_text = cleaned_text.lower()
        e = RegexpTokenizer(r'[a-zA-Z]+').tokenize(cleaned_text)
        filtered_sentence = [w for w in e if not w in self.builtinstopwords]
        return " ".join(filtered_sentence)

    def classify_email_sentiment(self, email_body):
        mail = self.preprocess(email_body)
        tfidf_v = TfidfVectorizer(vocabulary=self.vocabulary)  # Save Tfidf Vectorizer Instead
        X = tfidf_v.fit_transform([mail])

        temp_df = pd.DataFrame(self.model_instance.predict(X), columns=["tag"])
        temp_df["tag"] = temp_df["tag"].map(self.binaryReverseReplacement)
        temp_df["flag"] = np.amax(self.model_instance.predict_proba(X), axis=1)

        # output_df = pd.concat([df, temp_df], axis=1)
        # return (output_df.values.tolist())
        result = temp_df.loc[0,"tag"] + ',' + str(temp_df.loc[0,"flag"])
        return (result)