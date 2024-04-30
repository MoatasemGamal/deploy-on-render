import librosa
from tensorflow.keras.models import Sequential, model_from_json
import numpy as np


try:
    loaded_model
except NameError:
    json_file = open('CNN_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("best_model1_weights.h5")


import pickle

with open('scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)
print("Done")

def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)
def rmse(data,frame_length=2048,hop_length=512):
    rmse1=librosa.feature.rms(y=data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(rmse1)
def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    mfcc=librosa.feature.mfcc(y=data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

def extract_features(data,sr=22050,frame_length=2048,hop_length=512):
    result=np.array([])
    
    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result

def get_predict_feat(path):
    d, s_rate= librosa.load(path, duration=2.5, offset=0.6)
    res=extract_features(d)
    result=np.array(res)
    result=np.reshape(result,newshape=(1,2376))
    i_result = scaler2.transform(result)
    final_result=np.expand_dims(i_result, axis=2)
    
    return final_result



emotions1={1:'Neutral', 2:'Calm', 3:'Happy', 4:'Sad', 5:'Angry', 6:'Fear', 7:'Disgust',8:'Surprise'}
def prediction(path1):
    res=get_predict_feat(path1)
    predictions=loaded_model.predict(res)
    y_pred = np.argmax(predictions)
    return  y_pred



from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set maximum file size to 512 MB
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def list_files_in_folder(folder_path):
    if not os.path.isdir(folder_path):
        return {"error": f"The folder '{folder_path}' does not exist."}
    return os.listdir(folder_path)

@app.route('/predict', methods=['GET'])
def list_files():
    folder_path = "./uploads"  # Change this to your desired folder path
    files = list_files_in_folder(folder_path)
    return jsonify(files)

@app.route('/', methods=['GET'])
def hello():
    print("hello")

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No uploaded file'})

    # Check if the file format is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'})

    # Save the file to the upload folder
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Process the audio file using your model
    result = emotions1[prediction(filepath)]

    # Delete the uploaded file
    os.remove(filepath)

    return jsonify({'success':True, 'sentiment':result})

#if __name__ == '__main__':
#    app.run(debug=True)
