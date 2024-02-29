from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,pipeline
import math
model=AutoModelForSeq2SeqLM.from_pretrained('pegasus-samsum-model')
tokenizer=AutoTokenizer.from_pretrained('tokenizer')
def newsTitle(text):
    gen_kwargs={"length_penalty":0.8,"num_beams":8,"max_length":15}
    pipe=pipeline('summarization',model=model,tokenizer=tokenizer)
    res=pipe(text,**gen_kwargs)[0]["summary_text"]
    return res

if __name__=='__main__':
    print(newsTitle('''Once upon a time, a farmer had a goose that laid a golden egg every day. The farmer used to sell that egg and earn enough money to meet their family's day-to-day needs. One day, the farmer thought that if he could get more such golden eggs and make a lot of money and become a wealthy person. The farmer decided to cut the goose and remove all the golden eggs from its stomach. As soon as they killed the bird and opened the goose’s stomach, they found no eggs. The foolish farmer realized they had destroyed their last resource out of greed. '''))