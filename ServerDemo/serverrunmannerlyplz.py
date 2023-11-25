import json
from nltk.corpus import stopwords
import gensim
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import socket
import os
from collections import defaultdict
from gensim import corpora, models
from gensim.matutils import sparse2full
from sompy import SOMFactory
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('wordnet')

UDP_IP= "127.0.0.1"
UDP_PORT_O= 4243
UDP_PORT_I=4244

opened_socket_O = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

opened_socket_I = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
opened_socket_I.bind(("127.0.0.1", UDP_PORT_I))

json_full_dictionary = 'C:/Users/dongchen/Desktop/TD/TDBooks.json'
tfidf_matrix_reducedTD = 'C:/Users/dongchen/Desktop/TD/feature_matrixTD.npy'
svdModel = 'C:/Users/dongchen/Desktop/TD/svdTD.pkl'
SOM_Path = 'C:/Users/dongchen/Desktop/TD/SOMTD.pkl'
tfidf_vectorizerModel = "C:/Users/dongchen/Desktop/TD/tfidf_vectorizerTD.pkl"
SOM_PathBU = "C:/Users/dongchen/Desktop/bu/SOMBU.pkl"
svdModelBU = "C:/Users/dongchen/Desktop/bu/svdBU.pkl"
tfidf_matrix_reducedBU = "C:/Users/dongchen/Desktop/bu/feature_matrixBU.npy"
json_full_dictionaryBU = "C:/Users/dongchen/Desktop/bu/LibraryBU.json"
tfidf_vectorizerModelBU = "C:/Users/dongchen/Desktop/bu/tfidf_vectorizerBU.pkl"



with open(SOM_Path, 'rb') as file:
	SOM = pickle.load(file)

with open(SOM_PathBU, 'rb') as file:
	SOM_BU = pickle.load(file)

def main():
	print('ready')
	pid = os.getppid()
	#put_packet(["PIDs", pid])

	last_message = ""

	STOP = False
	while not STOP:
		p = get_packet()
		if p:
			if p == 'STOP':
				STOP = True
			if not p == last_message:
				#print(p)
				last_message = p
				o = Vectorize_new_documentBU(p)
				k = find_BMU_BU(o)
				v = Vectorize_new_document(p)
				i = find_BMU(v)
				put_packet([str(i[0])+'_'+str(i[1]), str(k[0])+'_'+str(k[1]), last_message])

def put_packet(packet):
	byte_message = bytes(json.dumps(packet), "utf-8")
	opened_socket_O.sendto(byte_message, (UDP_IP, UDP_PORT_O))

def get_packet():
	data, addr = opened_socket_I.recvfrom(1024)
	if data:
		return data.decode("utf-8")
	else:
		return None

def get_stemmed_doc(words):
	filtered = []
	for w in words:
		if w not in STOP_WORDS:
			stem = STEMMER.stem(w)
			if stem in ENGLISH_WORDS:
				filtered.append(stem)
			else:
				lemma = LEMMATIZER.lemmatize(w)
				if lemma in ENGLISH_WORDS:
					filtered.append(lemma)
	return filtered

ENGLISH_WORDS = set(nltk.corpus.words.words())
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))


def preprocess_new_document(sentence):
	text = gensim.utils.simple_preprocess(sentence, min_len = 3, deacc = True)
	text2 = get_stemmed_doc(text)
	text3 = " ".join(text2)
	return text3

def Vectorize_new_document(review):
	with open(tfidf_vectorizerModel, 'rb') as file:
		tfidf_vectorizer = pickle.load(file)
	processed_sentence = preprocess_new_document(review)
	query_vector = tfidf_vectorizer.transform([processed_sentence])

	with open(svdModel, 'rb') as f:
		svd = pickle.load(f)
	query_vector_reduced = svd.transform(query_vector)

	return query_vector_reduced

def Vectorize_new_documentBU(review):
	with open(tfidf_vectorizerModelBU, 'rb') as file:
		tfidf_vectorizerBU = pickle.load(file)
	processed_sentence = preprocess_new_document(review)
	query_vectorBU = tfidf_vectorizerBU.transform([processed_sentence])

	with open(svdModelBU, 'rb') as f:
		svdBU = pickle.load(f)
	query_vector_reducedBU = svdBU.transform(query_vectorBU)

	return query_vector_reducedBU


def find_BMU(v):
	bmus = SOM.project_data(v)
	xy = SOM.bmu_ind_to_xy(bmus)
	bmu_coordinate = xy[0][0:2]

	return bmu_coordinate

def find_BMU_BU(o):
	bmus = SOM_BU.project_data(o)
	xy = SOM_BU.bmu_ind_to_xy(bmus)
	bmu_coordinate_BU = xy[0][0:2]

	return bmu_coordinate_BU


main()