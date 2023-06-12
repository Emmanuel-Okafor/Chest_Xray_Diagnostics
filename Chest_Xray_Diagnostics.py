import  gradio as  gr
import  tensorflow as tf
import  numpy as  np
from tensorflow.keras.models import load_model


#Load the trained weights

model = load_model("COVIDDenseNet121_3_epochs.h5")
class_labels = [ 'NORMAL', 'PNEUMONIA']

# Deep Learning for Chest Xray  Prediction 

def Classify_lungs(img):
    img  = img.reshape((-1, 224, 224, 3))
    prediction = model.predict(img).flatten()
    return {class_labels[i]: float(prediction[i]) for i in range(2)}



#Initialization of the  input  components

img = gr.inputs.Image(shape=(224,224))
label = gr.outputs.Label(num_top_classes=2)



gr.Interface( 
    fn=Classify_lungs, inputs=img, outputs=label, 
    title="Chest X-Ray Diagnostic Application Tool",
    description="A diagnostic medical tool that predicts the existence of pneumonia in an X-ray image",
    
    interpretation="default"
).launch()
