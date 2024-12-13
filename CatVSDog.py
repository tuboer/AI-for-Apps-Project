#this is running on a local environment; if they wanted to have an app accesible to others, you need a cloud environment (i.e streamlit cloud)

import streamlit as st #importing library streamlit and renaming it to st, a standard python convention 
import json #deals with JSON formats - converting arbitrary data structres to string and back 
import requests #deals with responses and requests for http protocol 
import base64 #everything is stored as a byte, can convert ASCII (human readable format); e.g. 10 bytes -> 80 bits -> in 6 bit chunks -> each chunk to ASCII character; binary -> readable form in ASCII
from PIL import Image #pil is a standard python image library; manages image files
import io #stands for input and output; works with files  


#these are main classes your image is trained on
#you can define the classes in alphabectical order
PREDICTED_LABELS = {'Cats': 'cat', 'Dogs':'dog'} #categories -> what's on screen in the app 

def get_prediction(image_data): #expects parameter image data
  #replace your image classification ai service URL
  url = 'https://askai.aiclub.world/3e18a56f-5058-448e-8982-5ef61416c6d9'  
  r = requests.post(url, data=image_data) #data sent to the URL, requests library use
  #rest-style endpoint - post the data & get response back 
  print(f"response={r.json()}") #response itself is an object;  if you want the details of response, do r.json (understandable)
  response = r.json()['predicted_label'] #response back from AI 
  score = r.json()['score'] #score value & confidence level back from AI 
  #print("Predicted_label: {} and confidence_score: {}".format(response,score)) #debugging command 
  return response, score

#creating the web app

#setting up the title
st.title("Cat and Dog Image Classifier") 

#file uploader
image = st.file_uploader(label="Upload an image",accept_multiple_files=False, help="Upload an image to classify them")
#gives user option to upload a file, call image library, base 64 encoding, and display image
if image:
    #converting the image to bytes
    img = Image.open(image)
    buf = io.BytesIO() #takes image file and gets bytes out of it 
    img.save(buf,format = 'JPEG')
    byte_im = buf.getvalue()

    #converting bytes to b64encoding
    payload = base64.b64encode(byte_im) #take image and covert it to base64 encoded ASCII

    #setting up the image
    st.image(img)

    #predictions
    response, scores = get_prediction(payload) #call function get_prediction to get response

    #if you are using the model deployment in navigator
    #you need to define the labels
    response_label = PREDICTED_LABELS[response] #convert category name to user-friendly message (defined in dictionary in the start of code)

    col1, col2 = st.columns(2) #creating 2 columns to show this info; 1st: prediction label & 2nd: confidence score
    with col1:
      st.metric("Prediction Label",response_label)
    with col2:
      st.metric("Confidence Score", max(scores))
