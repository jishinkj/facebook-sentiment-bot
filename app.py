import os, sys
from flask import Flask, request
from pymessenger import Bot

from model_loader import ModelLoader
from sentiment import SeaModeling

import nltk
# nltk.download('punkt')

app = Flask(__name__)

PAGE_ACCESS_TOKEN = ""

bot = Bot(PAGE_ACCESS_TOKEN)

basedir = os.path.abspath(os.path.dirname(__file__))

if(os.name != 'posix'):
    SEA_MODEL_PATH=os.path.join(basedir, 'sources/sea_model_v1.pickle').replace('\\', '/')
    SEA_VOCABULARY_PATH =os.path.join(basedir, 'sources/sea_vocabulary_v1.pickle').replace('\\', '/')

else:
    SEA_MODEL_PATH =os.path.join(basedir, 'sources/sea_model_v1.pickle')
    SEA_VOCABULARY_PATH =os.path.join(basedir, 'sources/sea_vocabulary_v1.pickle')


@app.route('/', methods = ['GET'])
def verify():
	# webhook verification
	if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
		if not request.args.get("hub.verify_token") == "hello":
			return "Verification token mismatch", 403
		return request.args["hub.challenge"], 200
	return "Hello, World", 200

@app.route('/', methods = ['POST'])
def webhook():
	model_loader_obj = ModelLoader()
	sea_model_instance = model_loader_obj.get_sea_model()
	sea_model_vocab = model_loader_obj.get_sea_vocabulary()

	sea_obj = SeaModeling(sea_model_instance, sea_model_vocab)

	# print("========Model Loading Complete ==============")
	data = request.get_json()
	# log(data)

	if data['object'] == 'page':
		for entry in data['entry']:
			for messaging_event in entry['messaging']:

				# IDs
				sender_id = messaging_event['sender']['id']
				recipient_id = messaging_event['recipient']['id']

				if messaging_event.get('message'):
					if 'text' in messaging_event['message']:
						messaging_text = messaging_event['message']['text']

						result = sea_obj.classify_email_sentiment(messaging_text)
						#print("=====================================================")
				        #print(type(sea_output_df))
				        #print(len(sea_output_df))
				        #print(result)
				        #print("=====================================================")
					else:
						messaging_text = 'no text'

					# Echo
					#response = messaging_text

					#bot.send_text_message(sender_id, response)
					bot.send_text_message(sender_id, result)

	return "ok", 200

def log(message):
	print(message)
	sys.stdout.flush()

if __name__ == "__main__":
	# app.run(debug = True, port = 80)
	app.run(debug = True, port = 80)
