# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:05:39 2024

@author: Brain Hacker
"""

#%% Importing Libraries
import os
import google.generativeai as genai
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
import neattext.functions as nfx
import neattext as nt
from nltk.stem import WordNetLemmatizer
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import Chroma
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
#%% Setting up google API
os.environ['GOOGLE_API_KEY'] = ""
genai.configure(api_key = os.environ['GOOGLE_API_KEY'])
#%%
data = pd.read_csv("")
#%%
first_column_name = data.columns[0]
data.drop(first_column_name,axis=1,inplace=True)
data.dropna(inplace=True)
data.isna().sum()
data = data.dropna(subset=['text'])
data['text'] = data['text'].apply(lambda x: str(x) if not pd.isna(x) else '')
#%%
sentences = data['text'].tolist()
sentiment = data['sentiment'].tolist()
#%%
embeddings_model = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
embeddings = embeddings_model.embed_documents(sentences)
#%%
y = sentiment
X = embeddings
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state=99)
model= RandomForestClassifier()
result= model.fit(X, y)
y_pred= model.predict(X_test)
#%% Get and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
#%% Build the plot
plt.figure(figsize=(10,7))
sns.set(font_scale=1)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()
#%%

