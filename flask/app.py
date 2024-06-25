import os
import io
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Input, TFSMLayer
from tensorflow.keras.models import Model
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

tfs_layer = TFSMLayer("D:/Downloads/EffNetGemini30.pbmodel", call_endpoint='serving_default')

input_tensor = Input(shape=(224, 224, 3))
output_tensor = tfs_layer(input_tensor)
model = Model(input_tensor, output_tensor) 

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class_descriptions = {
    0: "The skin remains unbroken without any open sores or pre-ulcerative lesions. It is possible that there may be a deformity or cellulitis present.",
    1: "Superficial ulcer, also known as partial- or full-thickness ulcer, can be described as an ulceration that affects either the topmost layers of the skin or extends deeper into the underlying tissues.",
    2: "The deep ulcer has progressed to involve the ligament, tendon, joint capsule, bone, or deep fascia, without the presence of an abscess or osteomyelitis (OM).",
    3: "Conditions such as severe abscess, osteomyelitis, or septic arthritis",
    4: "Gangrene affecting a portion of the foot.",
    5: "Gangrene affecting the entire foot.",
}

def clear_uploads_folder():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file: {file_path}, {e}")

def preprocess_image(img):
    img = img.resize((224, 224))  
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  
    return img

clear_uploads_folder()

@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            
            image = Image.open(file)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            image.save(image_path) 

            image = Image.open(file)
            processed_image = preprocess_image(image)
            predictions = model(processed_image)
            probabilities = predictions["dense_1"][0].numpy()
            class_idx = np.argmax(probabilities)
            
            return render_template(
                'result.html', 
                class_label=f"Wagner Grade {class_idx}", 
                description=class_descriptions[class_idx],
                probability=f"{probabilities[class_idx]:.2f}",
                image_path=file.filename
            )

    return render_template('index.html')  

@app.route('/uploads/<filename>')

def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
