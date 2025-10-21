from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os

#create app
app = Flask(__name__)

#load the trained model
model = load_model('models/model.h5')

class_labels = ['pituitary','glioma', 'notumor','meningioma']

UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

#helper or predict tumor type

def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor Detected", confidence_score
    else:
        return f"Tumor Detected: {class_labels[predicted_class_index]}", confidence_score

@app.route('/', methods=['GET', 'POST'])
def index(): 
    if request.method == 'POST':           
        file = request.files['file']
        if file:
            file_location= os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            result, confidence = predict_tumor(file_location)


            return render_template('index.html', result=result, confidence=f'{confidence*100 : .2f}%', file_path = f'uploads/{file.filename}')
    return render_template('index.html', result=None) 
        
if __name__ == '__main__':
    app.run(debug=True)