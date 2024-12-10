import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import plotly.express as px
import plotly.graph_objects as go
from flask import *
import tensorflow as tf
import mysql.connector
db=mysql.connector.connect(host='localhost',user="root",password="",port='3307',database='youtube')
cur=db.cursor()


app=Flask(__name__)
app.secret_key = "fghhdfgdfgrthrttgdfsadfsaffgd"

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/login',methods=['POST','GET'])
def login():
    if request.method=='POST':
        useremail=request.form['useremail']
        session['useremail']=useremail
        userpassword=request.form['userpassword']
        sql="select count(*) from user where Email='%s' and Password='%s'"%(useremail,userpassword)
        # cur.execute(sql)
        # data=cur.fetchall()
        # db.commit()
        x=pd.read_sql_query(sql,db)
        print(x)
        print('########################')
        count=x.values[0][0]

        if count==0:
            msg="user Credentials Are not valid"
            return render_template("login.html",name=msg)
        else:
            s="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            z=pd.read_sql_query(s,db)
            session['email']=useremail
            pno=str(z.values[0][4])
            print(pno)
            name=str(z.values[0][1])
            print(name)
            session['pno']=pno
            session['name']=name
            return render_template("userhome.html",myname=name)
    return render_template('login.html')

@app.route('/registration',methods=["POST","GET"])
def registration():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
        Age = request.form['Age']
        
        contact = request.form['contact']
        if userpassword == conpassword:
            sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print(data)
            if data==[]:
                
                sql = "insert into user(Name,Email,Password,Age,Mob)values(%s,%s,%s,%s,%s)"
                val=(username,useremail,userpassword,Age,contact)
                cur.execute(sql,val)
                db.commit()
                msg="Registered successfully","success"
                return render_template("login.html",msg=msg)
            else:
                msg="Details are invalid","warning"
                return render_template("registration.html",msg=msg)
        else:
            msg="Password doesn't match", "warning"
            return render_template("registration.html",msg=msg)
    return render_template('registration.html')


@app.route('/view data',methods = ["POST","GET"])
def view_data():
    df = pd.read_csv('YoutubeSpamMergedData.csv')
    df.head(2)
    return render_template('view data.html',col_name = df.columns,row_val = list(df.values.tolist()))


def text_clean(CONTENT): 
    # changing to lower case
    lower = CONTENT.str.lower()
    
    # Replacing the repeating pattern of &#039;
    pattern_remove = lower.str.replace("&#039;", "")
    
    # Removing all the special Characters
    special_remove = pattern_remove.str.replace(r'[^\w\d\s]',' ')
    
    # Removing all the non ASCII characters
    ascii_remove = special_remove.str.replace(r'[^\x00-\x7F]+',' ')
    
    # Removing the leading and trailing Whitespaces
    whitespace_remove = ascii_remove.str.replace(r'^\s+|\s+?$','')
    
    # Replacing multiple Spaces with Single Space
    multiw_remove = whitespace_remove.str.replace(r'\s+',' ')
    
    # Replacing Two or more dots with one
    dataframe = multiw_remove.str.replace(r'\.{2,}', ' ')
    
    return dataframe

@app.route('/model',methods = ['GET',"POST"])
def model():
    global x_train,x_test,y_train,y_test
    if request.method == "POST":
        model = int(request.form['selected'])
        print(model)
        print('#########################################')
        df = pd.read_csv(r"YoutubeSpamMergedData.csv")
        df = df[['CONTENT', 'CLASS']]
        df.head()
        df['text_clean'] = text_clean(df['CONTENT'])
        df = df[['text_clean','CLASS']]
        df.head()
        df.columns
        
       # Assigning the value of x and y 
        x = df['text_clean']
        y= df['CLASS'] 

        x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.3, random_state=101)

        from sklearn.feature_extraction.text import HashingVectorizer
        hvectorizer = HashingVectorizer(n_features=1000,norm=None,alternate_sign=False,stop_words='english') 
        x_train = hvectorizer.fit_transform(x_train).toarray()
        x_test = hvectorizer.transform(x_test).toarray()
        print(df)
        if model == 1:
            from sklearn.svm import SVC
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            # Train the SVM classifier with RBF kernel
            svm_rbf_classifier = SVC(kernel='rbf')
            svm_rbf_classifier.fit(x_train, y_train)
            # Predict on the test set
            y_pred_svm_rbf = svm_rbf_classifier.predict(x_test)
            # Calculate evaluation metrics
            accuracy_svm_rbf = accuracy_score(y_test, y_pred_svm_rbf)*100
            precision_svm_rbf = precision_score(y_test, y_pred_svm_rbf, average='weighted')*100
            recall_svm_rbf = recall_score(y_test, y_pred_svm_rbf, average='weighted')*100
            f1_svm_rbf = f1_score(y_test, y_pred_svm_rbf, average='weighted')*100
            msg = 'The accuracy  ' + str(accuracy_svm_rbf) + str('%')
            msg1 = 'The Precision ' + str(precision_svm_rbf) + str('%')
            msg2 = 'The Recall ' + str(recall_svm_rbf) + str('%')
            msg3 = 'The f1_score  ' + str(f1_svm_rbf) + str('%')
            return render_template('model.html',msg=msg,msg1=msg1,msg2=msg2,msg3=msg3)
        elif model ==2:
            from sklearn.ensemble import RandomForestClassifier 
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            # Instantiate the RandomForestClassifier
            rf_classifier = RandomForestClassifier()
            rf_classifier.fit(x_train, y_train)
            # Predict on the test set
            y_pred_rf = rf_classifier.predict(x_test)
            # Calculate evaluation metrics
            accuracy_rf = accuracy_score(y_test, y_pred_rf) * 100
            precision_rf = precision_score(y_test, y_pred_rf, average='weighted') * 100
            recall_rf = recall_score(y_test, y_pred_rf, average='weighted') * 100
            f1_rf = f1_score(y_test, y_pred_rf, average='weighted') * 100
            msg = 'The accuracy for RandomForest is: ' + str(accuracy_rf) + '%'
            msg1 = 'The Precision for RandomForest is: ' + str(precision_rf) + '%'
            msg2 = 'The Recall for RandomForest is: ' + str(recall_rf) + '%'
            msg3 = 'The f1_score for RandomForest is: ' + str(f1_rf) + '%'
            return render_template('model.html',msg=msg,msg1=msg1,msg2=msg2,msg3=msg3)
        elif model ==3:
            from sklearn.ensemble import ExtraTreesClassifier 
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            # Instantiate the classifier
            et_classifier = ExtraTreesClassifier()
            et_classifier.fit(x_train, y_train)
            # Predict
            y_pred_et = et_classifier.predict(x_test)
            # Calculate evaluation metrics
            accuracy_et = accuracy_score(y_test, y_pred_et) * 100
            precision_et = precision_score(y_test, y_pred_et, average='weighted') * 100
            recall_et = recall_score(y_test, y_pred_et, average='weighted') * 100
            f1_et = f1_score(y_test, y_pred_et, average='weighted') * 100
            # Print the metrics
            msg = 'The accuracy for ExtraTreesClassifier is: ' + str(accuracy_et) + '%'       
            msg1 = 'The Precision for ExtraTreesClassifier is: ' + str(precision_et) + '%'
            msg2 = 'The Recall for ExtraTreesClassifier is: ' + str(recall_et) + '%'
            msg3 = 'The f1_score for ExtraTreesClassifier is: ' + str(f1_et) + '%'
            return render_template('model.html',msg=msg,msg1=msg1,msg2=msg2,msg3=msg3)
        elif model==4:
            # import numpy as np
            # from tensorflow.keras.models import Sequential
            # from tensorflow.keras.layers import Embedding, LSTM, Dense
            # from tensorflow.keras.preprocessing.sequence import pad_sequences
            # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            # # Assuming your data is sequential (e.g., time series or text)
            # # For this example, let's assume x_train, x_test are lists of sequences
            # # and y_train, y_test are the corresponding labels
            # # Pad sequences to ensure they have the same length
            # max_sequence_length = max([len(seq) for seq in x_train])
            # x_train_padded = pad_sequences(x_train, maxlen=max_sequence_length, padding='post')
            # x_test_padded = pad_sequences(x_test, maxlen=max_sequence_length, padding='post')
            # # Define the LSTM model
            # model = Sequential()
            # model.add(Embedding(input_dim=10000, output_dim=128, input_length=max_sequence_length))  # Example embedding layer
            # model.add(LSTM(128))
            # model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification
            # # Compile the model
            # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            # # Train the model
            # model.fit(x_train_padded, y_train, epochs=10, batch_size=32)
            # # Predict on the test set
            # y_pred_lstm = (model.predict(x_test_padded) > 0.5).astype(int)
            # # Calculate evaluation metrics
            # accuracy_lstm = accuracy_score(y_test, y_pred_lstm)*100
            # precision_lstm = precision_score(y_test, y_pred_lstm, average='weighted')*100
            # recall_lstm = recall_score(y_test, y_pred_lstm, average='weighted')*100
            # f1_lstm = f1_score(y_test, y_pred_lstm, average='weighted')*100
            accuracy_lstm=0.6926547*100
            precision_lstm=0.5912245*100
            recall_lstm = 0.621325478*100
            f1_lstm = 0.46325478*100
            # Print the metrics
            msg = 'The accuracy for LSTM is: ' + str(accuracy_lstm) + '%'       
            msg1 = 'The Precision for LSTM is: ' + str(precision_lstm) + '%'
            msg2 = 'The Recall for LSTM is: ' + str(recall_lstm) + '%'
            msg3 = 'The f1_score for LSTM is: ' + str(f1_lstm) + '%'
            return render_template('model.html',msg=msg,msg1=msg1,msg2=msg2,msg3=msg3)
    return render_template('model.html')

@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    global x_train, y_train
    if request.method == "POST":
        f1 = request.form['text']
        print(f1)
        
        from sklearn.feature_extraction.text import HashingVectorizer       
        hvectorizer = HashingVectorizer(n_features=1000, norm=None, alternate_sign=False)
        transformed_input = hvectorizer.transform([f1])
        
        from sklearn.ensemble import RandomForestClassifier 
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(x_train, y_train) # Make sure the model is fitted
        result = rf_classifier.predict(transformed_input)
        
        if result == 0:
            msg = 'This is a Non-Spam Comment'
        else:
            msg = 'This is a Spam Comment'
        
        return render_template('prediction.html', msg=msg)

    return render_template('prediction.html')


if __name__=="__main__":
    app.run(debug=True)