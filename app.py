import os
import numpy as np
# import pickle
# # Tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input


# path_to_train = "D:\dataset\separate\\apple\Test"


# Define a flask app
app = Flask(__name__)

# Load Labels
# labels = pickle.load(open('labels.pkl','rb'))

# # Model saved with Keras model.save()
model = load_model("resnet50.h5")


def model_predict(img_path, model):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(180, 180))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img, axis=0)
    preprocessed_img = tf.keras.applications.resnet.preprocess_input(img_array)

    class_names=['Blotch_Apple', 'Normal_Apple', 'Rot_Apple', 'Scab_Apple']

    # Perform prediction
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]


        # Map predicted class label to treatment and pesticide
    class_to_treatment = {
        0: 'Apply a suitable fungicide spray containing captan or mancozeb. Prune infected branches and destroy fallen leaves.',
        1: 'No specific treatment needed. Maintain good cultural practices, such as proper watering and fertilization.',
        2: 'Discard infected apples. Maintain proper storage conditions with good ventilation and controlled humidity.',
        3: 'Apply a suitable fungicide spray containing myclobutanil or sulfur. Prune and destroy infected plant parts.'
    }

    class_to_pesticide = {
        0: 'Use fungicides containing captan or mancozeb at recommended doses.',
        1: 'No specific pesticide needed.',
        2: 'Maintain proper storage conditions, including temperature and humidity control. No specific pesticide recommended.',
        3: 'Use fungicides containing myclobutanil or sulfur as per label instructions.'
    }

    result_dict = {
        "Class": predicted_class,
        "Treatment": class_to_treatment[predicted_class_index],
        "Pesticide": class_to_pesticide[predicted_class_index]
    }
    print( result_dict)
    return result_dict
    # return result_string
#    user_img = image.load_img(img_path,target_size = (100,100))
#    input_arr = keras.utils.img_to_array(user_img)

#    input_arr = np.array([input_arr])  # Convert single image to a batch.
#    predictions = model.predict(input_arr)
#    preds_idx = np.argmax(predictions, axis=1)


#    dict = {'fruit_name':'apple','fruit_disease':os.listdir(path_to_train)[preds_idx[0]]}
#    print("Detected Disease is: ",os.listdir(path_to_train)[preds_idx[0]])

#    print(model.summary())

   #  img = image.load_img(img_path, target_size=(224, 224))

   #  # Preprocessing the image
   #  x = image.img_to_array(img)
   #  ## Scaling
   #  x = x/255
   #  x = np.expand_dims(x, axis=0)

   #  preds = model.predict(x)
   #  preds_idx = np.argmax(preds, axis=1)
   #  if preds_idx==0:
   #      preds_idx="The leaf is healthy"
   #  elif preds_idx==1:
   #      preds_idx="The leaf has Multiple Disease"
   #  elif preds_idx==2:
   #      preds_idx="The leaf is infected with Rust"
   #  else:
   #      preds_idx="The leaf has Scab"
        
   


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')



@app.route('/aboutus', methods=['GET'])
def aboutus():
    # Main page
    return render_template('about.html')



@app.route('/check_disease', methods=['GET'])
def check_disease():
    # Main page
    return render_template('upload_file.html')


@app.route('/result', methods=['GET','POST'])
def result():
   if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

      #   # Make prediction
        preds = model_predict(file_path, model)
        print(preds)
   return render_template('result.html',data=preds)


# @app.route('/predict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         # Get the file from post request
#         f = request.files['file']

#         # Save the file to ./uploads
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)

#         # Make prediction
#         preds = model_predict(file_path, model)
#         print(preds)
#         return preds
#     return None


if __name__ == '__main__':
    app.run(debug=True)
