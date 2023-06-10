import json
import pickle


from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
with open('decision_tree.pkl', 'rb') as f:
    model = pickle.load(f)
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/lrmodel')
def lrmodel():
    return render_template('lrmodel.html')

@app.route('/dtmodel')
def dtmodel():
    return render_template('dtmodel.html')

@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("lrresults.html",prediction_text="The AC power prediction using Linear Regression is {} kW".format(output))

@app.route('/predictdf',methods=['POST'])
def predictdf():
    data1=[float(x) for x in request.form.values()]
    final_input1=scalar.transform(np.array(data1).reshape(1,-1))
    print(final_input1)
    df=model.predict(final_input1)[0]
    return render_template("dtresults.html",prediction_text1="The AC power prediction using Decision Tree is {} kW".format(df))


if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0',port=8080)

#if __name__ =='__main__':
 #  app.run(host='0.0.0.0',port=8080) 
     
