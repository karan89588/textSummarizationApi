from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,pipeline
import math
model=AutoModelForSeq2SeqLM.from_pretrained('pegasus-samsum-model')
tokenizer=AutoTokenizer.from_pretrained('tokenizer')

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from heapq import nlargest
from string import punctuation

def generateSummary(text):
	gen_kwargs={"length_penalty":0.8,"num_beams":8,"max_length":128}
	pipe=pipeline('summarization',model=model,tokenizer=tokenizer)
	l=0
	select=0.15
	length=len(text.split())
	print(length)
	if(length>10000):
		select=0.1
	if(length>15000):
		select=0.05
	if(length<500):
		gen_kwargs['max_length']=int(length*0.5)
	elif(length<1000):
		gen_kwargs['max_length']=int(length*0.4)
	elif(length<1500):
		gen_kwargs['max_length']=int(length*0.3)
	elif(length<2000):
		gen_kwargs['max_length']=int(length*0.2)
	else:
		gen_kwargs['max_length']=128
	if(length>2000):
		stopwords=list(STOP_WORDS)
		nlp=spacy.load('en_core_web_sm')
		doc=nlp(text)
		tokens=[token.text for token in doc]
		print(punctuation)
		word_frequencies={}
		for word in doc:
			if word.text.lower() not in stopwords:
				if word.text.lower() not in punctuation:
					if word.text not in word_frequencies.keys():
						word_frequencies[word.text]=1
					else:
						word_frequencies[word.text]+=1

		max_frequency=max(word_frequencies.values())
		for word in word_frequencies.keys():
			word_frequencies[word]=word_frequencies[word]/max_frequency
		sentence_tokens=[sent for sent in doc.sents]
		sentence_scores={}
		for sent in sentence_tokens:
			for word in sent:
				if word.text.lower() in word_frequencies.keys():
					if sent not in sentence_scores.keys():
						sentence_scores[sent]=word_frequencies[word.text.lower()]
					else:
						sentence_scores[sent]+=word_frequencies[word.text.lower()]
		select_length=int(len(sentence_tokens)*select)
		summary=nlargest(select_length,sentence_scores,key=sentence_scores.get)
		final_summary=[word.text for word in summary]
		extract_summary=' '.join(final_summary)
		text=extract_summary		

	summary=[]
	length=len(text.split())
	s=0
	t=math.ceil(length/600)
	print('Length of original sentence : ',length)
	print('Max Lenght Assigned for summary : ',gen_kwargs['max_length'])
	while(l<length):
		s+=1
		text1=" ".join(text.split()[l:l+600])
		summary.append(pipe(text1,**gen_kwargs)[0]["summary_text"])
		print('step ',s,' out of ',t)
		l+=600
	res=" ".join(summary)
	print('Length of Summary',len(res.split()))
	res=res.replace('<n>','\n')
	print(res)
	return res
if __name__=="__main__":

	text='''Google LLC (/ˈɡuːɡəl/ ⓘ GOO-ghəl) is an American multinational
	  technology company focusing on artificial intelligence,[9] online advertising,
	    search engine technology, cloud computing, computer software, quantum 
		computing, e-commerce, and consumer electronics. It has been referred
		to as "the most powerful company in the world"[10] and as one of the
		  world's most valuable brands due to its market dominance, data collection,
		    and technological advantages in the field of artificial intelligence.[11][12][13]
			  Google's parent company Alphabet Inc. is one of the five Big Tech companies,
			    alongside Amazon, Apple, Meta, and Microsoft.'''
	print(generateSummary(text))