from flask import Flask, jsonify, request,render_template
from flask_material import Material
import numpy as np
from sklearn.externals import joblib
import pickle
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib
from scipy.sparse import hstack
from bs4 import BeautifulSoup
import re
import string
from sklearn.feature_extraction.text import CountVectorizer


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)  #Flask constructor takes the name of current_Module
Material(app)


###################################################
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])


def clean_text(sentence):
    sentence = re.sub(r"http\S+", "", sentence)
    sentence = BeautifulSoup(sentence, 'lxml').get_text()
    sentence = decontracted(sentence)
    sentence = re.sub("\S*\d\S*", "", sentence).strip()
    sentence = re.sub('[^A-Za-z]+', ' ', sentence)
    # https://gist.github.com/sebleier/554280
    sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords)
    return sentence.strip()
###################################################
def get_length(s):
    return len(s),np.array([len(s)])

def get_Words_Count(s):
    return len(s.split(" ")),np.array([len(s.split(" "))])

def get_StopWords_Count(s):
    count=0
    for word in s.split(" "):
        if word in stopwords:
            count=count+1
    return count,np.array([count])

def get_UpperCase_Count(s):
    count=0
    for char in s:
        if ord(char)>=65 and ord(char)<=90:
            count=count+1
    return count,np.array([count])

def get_BadCount_Ratio(s):
    count=0
    bad_words = pd.read_csv('words_bad.txt', sep=",", header=None)
    words=bad_words.columns
    bad_words_arr=bad_words.iloc[0].tolist()
    final_bad_words=[]
    for word in bad_words_arr:
        word=word.strip() #Removing extra spaces from the words
        final_bad_words.append(word)
    for word in s.split(" "):
        if word in final_bad_words:
            count=count+1
    tot_words,tot=get_Words_Count(s)
    return count,np.array([count])

def get_Unique_Count(s):
    tot_words,tot=get_Words_Count(s)
    return len(set(s.split(" "))),np.array([len(set(s.split(" ")))]),(len(set(s.split(" ")))/tot_words)

def get_Punctuation_Count(s):
    count=0
    for ch in s:
        if ch in string.punctuation:
            count=count+1
    return count,np.array(count)

def get_Prediction(model,vectorizer):
        clf = pickle.load(open(model,'rb'))
        tfidf_vect = pickle.load(open(vectorizer,'rb'))
        to_predict_list = request.form.to_dict()
        review_text =(to_predict_list['question_text'])
        question=review_text
        length_t,length_text=get_length(review_text)
        tot_words,total_words=get_Words_Count(review_text)
        unique_count,unique_cnt,unique_cnt_ratio=get_Unique_Count(review_text)
        tot_stopwords,total_stopwords=get_StopWords_Count(review_text)
        upp_cnt,Uppercase_cnt=get_UpperCase_Count(review_text)
        punct_count,punct_cnt=get_Punctuation_Count(review_text)
        review_text = clean_text(to_predict_list['question_text'])
        tfidf_words=tfidf_vect.transform([review_text])
        bad_count,bad_cnt=get_BadCount_Ratio(review_text)
        bad_cnt_ratio=bad_count/tot_words
        punct_cnt=np.array([punct_cnt])
        data_m=hstack((tfidf_words,length_text,total_words,total_stopwords,Uppercase_cnt,bad_cnt_ratio,unique_cnt_ratio,punct_cnt,unique_cnt,bad_cnt)).tocsr()
        pred = clf.predict(data_m)
        print(pred[0])
        if pred[0]:
            prediction = "InSincere Question"
        else:
            prediction = "Sincere Question"
        word_list=review_text.split(" ")
        data_dict={}
        data_dict['Length']=length_t
        data_dict['Total Words']=tot_words
        data_dict['Total StopWords']=tot_stopwords
        data_dict['Uppercase Count']=upp_cnt
        data_dict['BadWords Count']=bad_count
        data_dict['BadWords ratio']=bad_cnt_ratio
        data_dict['Punctuations Count']=punct_count
        data_dict['Unique Words Count']=unique_count
        data_dict['Unique Words Ratio']=unique_cnt_ratio
        if vectorizer == "tfidf_vec.p":
            tfidf_features=tfidf_vect.get_feature_names()
            tfidf_idf_=tfidf_vect.idf_
            word_zip=dict(zip(tfidf_features,tfidf_idf_))
            print("In TFIDF")
            for word in word_list:
                print("In For")
                if word in word_zip.keys():
                    data_dict[word]=word_zip[word]
        data_dict['Question']=question
        data_dict['Prediction']=prediction
        return data_dict

@app.route('/')  #route() function is a decorator which tells the web Application which URL should call the associated
def hello_world():
    return 'Hello World!'

@app.route('/data_analysis')
def data_analyze():
    return render_template('data_analysis.html')

@app.route('/index')
def index():
    return flask.render_template('Testing_f.html')

@app.route('/compare')
def compare():
    return render_template('compare.html')

@app.route('/view_dataset')
def read_dataset():
    df=pd.read_csv('sample_train.csv')
    return render_template('view_dataset.html',data_view=df)

@app.route('/model_info')
def model_info():
    return render_template('model_info.html')
@app.route('/help')
def help():
    return render_template('help.html')


@app.route('/predict', methods=['POST'])
def predict():
    algo_selected=request.form['group1']
    text_vectorizer=request.form['group2']
    print(algo_selected)
    model_data={}
    if algo_selected=="Naive Bayes" and text_vectorizer=="TFIDF":
        data=get_Prediction('best_nb_tfidf.p','tfidf_vec.p')
    elif algo_selected=="Logistic Regression" and text_vectorizer=="TFIDF":
        data=get_Prediction('model_lr_f_best.p','tfidf_vec.p')   
    elif algo_selected=="Naive Bayes" and text_vectorizer=="BagOfWords":
        data=get_Prediction('model_nb_count_best.p','count_vec.p')
    elif algo_selected=="SVM" and text_vectorizer =="BagOfWords":
        data=get_Prediction('model_sv_f.p','count_vec.p')
    elif algo_selected=="SVM" and text_vectorizer=="TFIDF":
        data=get_Prediction('model_sv_f.p','tfidf_vec.p')
    else:
        data=get_Prediction('model_lr_best_count.p','count_vec.p')
    return render_template('prediction.html',result=data)

@app.route('/predict_2', methods=['POST'])
def predict_2():
    if request.method == 'POST':
        print("HELLo")
        algo_selected=request.form['group1']
        vec1=request.form.getlist('BOW')
        vec2=request.form.getlist('TFIDF')
        print(algo_selected)
        if algo_selected =="Naive Bayes":
            data_tfidf=get_Prediction('best_nb_tfidf.p','tfidf_vec.p')
            data_cnt=get_Prediction('best_nb_tfidf.p','count_vec.p')
            data_tfidf['Text Vectorizer']='TFIDF'
            data_tfidf['alpha']=1
            data_tfidf['fit_params']='None'
            data_tfidf['scoring']='f1'
            data_tfidf['fit_prior']='true'
            data_tfidf['iid']='true'

            data_cnt['Text Vectorizer']='BOW'
            data_cnt['alpha']=1
            data_cnt['fit_params']='None'
            data_cnt['scoring']='f1'
            data_cnt['fit_prior']='true'
            data_cnt['iid']='true'
            return render_template('prediction_2.html',result=data_tfidf,result1=data_cnt)
        elif algo_selected == "Logistic Regression":
            data_tfidf=get_Prediction('model_lr_f_best.p','tfidf_vec.p')
            data_cnt=get_Prediction('model_lr_best_count.p','count_vec.p')
            data_tfidf['alpha']='0.00001'
            data_tfidf['loss']='log'
            data_tfidf['class_weight']='balanced'
            data_tfidf['epsilon']=0.1
            data_tfidf['learning_rate']='optimal'
            data_tfidf['penalty']='l2'
            data_tfidf['fit_intercept']='True'
            data_tfidf['Text vectorizer']='TFIDF'

            data_cnt['Text Vectorizer']='BOW'
            data_cnt['alpha']='0.00001'
            data_cnt['loss']='log'
            data_cnt['class_weight']='balanced'
            data_cnt['epsilon']=0.01
            data_cnt['learning_rate']='optimal'
            data_cnt['penalty']='l1'
            data_cnt['fit_intercept']='True'
            return render_template('prediction_2.html',result=data_tfidf,result1=data_cnt)
        else:
            data_cnt = get_Prediction('model_sv_f.p','count_vec.p')
            data_tfidf = get_Prediction('model_sv_f.p','count_vec.p')
            data_tfidf['alpha']='0.0001'
            data_tfidf['loss']='hinge'
            data_tfidf['class_weight']='balanced'
            data_tfidf['epsilon']=0.1
            data_tfidf['learning_rate']='optimal'
            data_tfidf['penalty']='l2'
            data_tfidf['fit_intercept']='True'
            data_tfidf['Text vectorizer']='TFIDF'

            data_cnt['alpha']='0.0001'
            data_cnt['loss']='hinge'
            data_cnt['class_weight']='balanced'
            data_cnt['epsilon']=0.01
            data_cnt['learning_rate']='optimal'
            data_cnt['penalty']='l1'
            data_cnt['fit_intercept']='True'
            data_cnt['Text Vectorizer']='BOW' 
            return render_template('prediction_2.html',result=data_tfidf,result1=data_cnt)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
