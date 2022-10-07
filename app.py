import numpy as np
import model
from flask import Flask, request, render_template
import pickle

app = Flask(__name__,template_folder="templates")
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def predict():
    
    gre = request.args.get('gre')
    toefl = request.args.get('toefl')
    uni_rating = request.args.get('uni_rating')
    sop = request.args.get('sop')
    lor = request.args.get('lor')
    cgpa = request.args.get('cgpa')
    res = request.args.get('research')
    if(res=='Yes' or res=='yes' or res=='YES' or res=='y' or res=='Y' or res=='yES' or res=='YEs' or res=='yEs' or res=='YeS'):
        research = '1'
    else:
        research = '0'
    arr = np.array([gre, toefl, uni_rating, sop, lor, cgpa, research])
    brr = np.asarray(arr, dtype=float)
    output = model.predict([brr])
    out = output[0]*100
    return render_template('out.html', output=out)

if __name__ == "__main__":
    app.run(debug=True)