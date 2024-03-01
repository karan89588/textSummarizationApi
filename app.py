from flask import Flask, request, jsonify
from flask_cors import CORS
from getSum import generateSummary
from newsTitleGeneration import newsTitle
from deep_translator import GoogleTranslator
from langdetect import detect
from extractiveSum import generateExtractiveSummary
app=Flask(__name__)
CORS(app)
@app.route('/',methods=['GET'])
def get():
	return jsonify({'msg':'Hello'})

@app.route('/getSummary',methods=['POST'])
def getRes():
	try:
		txt=request.json['txt']
		target_lang=request.json['target_lang']
		user_demand=request.json['user_demand']
		if target_lang=='check':
			try:
				target_lang=detect(txt)
				if target_lang!='en':
					
					txt=GoogleTranslator(source='auto',target='en').translate(txt)
			except:
				return jsonify({'msg':'Oppps Language Not allowed','success':False})
		print(txt)
		output=generateSummary(txt)
		if user_demand!='':
			target_lang=user_demand
		if target_lang!='en':
			output=GoogleTranslator(source='auto',target=target_lang).translate(output)
		return jsonify({'msg':'success','success':True,'sum':output})
	except:
		return jsonify({'msg':'failed','success':False})
@app.route('/getTitle',methods=['POST'])
def getTitle():
	txt=request.json['txt']
	print(txt)
	output=newsTitle(txt)
	return jsonify({'msg':'success','success':True,'sum':output})
	
if __name__=='__main__':
	app.run(debug=True,port=4000)