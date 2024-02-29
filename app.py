from flask import Flask, request, jsonify
from flask_cors import CORS
from getSum import generateSummary
from newsTitleGeneration import newsTitle
app=Flask(__name__)
CORS(app)
@app.route('/',methods=['GET'])
def get():
	return jsonify({'msg':'Hello'})

@app.route('/getSummary',methods=['POST'])
def getRes():
	txt=request.json['txt']
	print(txt)
	output=generateSummary(txt)
	return jsonify({'msg':'success','success':True,'sum':output})

@app.route('/getTitle',methods=['POST'])
def getTitle():
	txt=request.json['txt']
	print(txt)
	output=newsTitle(txt)
	return jsonify({'msg':'success','success':True,'sum':output})
	
if __name__=='__main__':
	app.run(debug=True,port=4000)