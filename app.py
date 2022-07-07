# importing necessary libraries
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import pickle
import warnings
import pandas as pd
import matplotlib.pyplot as plt                                # styling pandas dataframe is depending on matplotlib
import os

warnings.filterwarnings("ignore")

IMG_SIZE = 224   # shape of the image


def predict_image(image_arr, img_size, model, class_map):
    """Function predicts bounding box co-ordinated and class of car model make

    parameters
    ----------
    image_arr : numpy.ndarray , image in numpy array format
    img_size  : int , to what size image has to be resized for model
    model     : tensorflow model , loaded tensorflow model
    class_map : dict, representing the class name map to its class number

    returns
    -------
    output_class_name : str , representing the predicted car model name
    df : pandas.DataFrame, representing top five predictions with confidence of each
    class_confidence : float, confidence of top predicted class
    bbox : numpy.array, representing the predicted bounding box co-ordinates
    """
    # if there is any grey image uploaded then it should be converted into rgb
    if image_arr.shape[-1] in [1, 2, 4]:
        image_arr = tf.image.grayscale_to_rgb(image_arr)

    # resizing the image to 224x224x3
    image = tf.image.resize(image_arr, [img_size, img_size]).numpy()
    image = np.expand_dims(image, axis=0)

    # predicting the class and bounding box
    bbox, class_info = model.predict(image)
    output_class_name = class_map[np.argmax(class_info)]
    class_confidence = class_info[0].max()

    # getting top five classes and converting it into dataframe to display
    top_5_scores_idx = np.argsort(class_info[0])[::-1][0:5]
    top_5_scores = class_info[0][top_5_scores_idx]
    top_5_names = [class_map[score] for score in top_5_scores_idx]
    df = pd.DataFrame()
    df['Car Name'] = top_5_names
    df['Confidence Level'] = top_5_scores

    # reshaping the bounding box from 224x224 size to actual height x width of uploaded image
    x, y, w, h = bbox[0]
    height_factor = image_arr.shape[0] / IMG_SIZE
    width_factor = image_arr.shape[1] / IMG_SIZE

    x, w = x * width_factor, w * width_factor
    y, h = y * height_factor, h * height_factor

    return output_class_name, df, class_confidence, np.array([x, y, w, h]).astype(int)


def image_with_bbox(image_arr, bounding_box):
    """Function returns an image with bounding box applied on it

    parameters
    ----------
    image_arr : numpy.ndarray , image in numpy array format
    bounding_box : numpy.array, representing the predicted bounding box co-ordinates

    returns
    -------
    image_arr : numpy.ndarray , modified image in numpy array format
    """
    # unpacking the predicted bounding box
    x, y, w, h = bounding_box

    # adding bounding box to image and returning it
    image_arr = cv2.rectangle(image_arr, (x, y), (x + w, y + h), (0, 255, 0), 4)

    return image_arr


# This function is used for metric for model. This is needed for loading the model
def IOU(y_true, y_pred):
    """Function ruturns the Intersection Over Union(IOU)

    parameters
    ----------
    y_true : numpy.ndarray, representing the ground truth bounding box of shape (n,4)
    y_pred : numpy.ndarray, representing the predicted bounding box of shape (n,4)

    returns
    -------
    iou   : float, representing the IOU value for given batch of values
    """
    # unpacking the columns of each co-ordinated
    x1, y1, w, h = y_true[:, 0], y_true[:, 1], y_true[:, 2], y_true[:, 3]
    x1_pred, y1_pred, w_pred, h_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]

    # finding intersection area
    diff_width = np.minimum(x1 + w, x1_pred + w_pred) - np.maximum(x1, x1_pred)
    diff_height = np.minimum(y1 + h, y1_pred + h_pred) - np.maximum(y1, y1_pred)
    intersection = diff_width * diff_height

    # individual area of bounding box of ground truth and predicted
    area_true = w * h
    area_pred = w_pred * h_pred

    union = area_true + area_pred - intersection

    intersection_sum = 0
    union_sum = 0

    # Compute intersection and union over multiple boxes
    for j, _ in enumerate(union):
        if union[j] > 0 and intersection[j] > 0 and union[j] >= intersection[j]:
            intersection_sum += intersection[j]
            union_sum += union[j]

    # Compute IOU. Use epsilon to prevent division by zero
    iou = np.round(intersection_sum / (union_sum + tf.keras.backend.epsilon()), 4)
    # This must match the type used in py_func
    iou = iou.astype(np.float32)
    return iou


def IoU(y_true, y_pred):
    iou = tf.py_function(IOU, [y_true, y_pred], Tout=tf.float32)
    return iou


# this method is cached so that model should load only once
@st.cache(allow_output_mutation=True)
def load_the_pretrained_model():
    # loading the tensorflow saved model
    model = tf.keras.models.load_model(r'./model/mobile_net_combi_with_aug_1_10-acc_0.8687-IoU_0.8758',
                                       custom_objects={'IoU': IoU})

    # loading the class label maps with integer class
    class_names = pickle.load(open(r'./model/class_names.pickle', "rb"))

    return model, class_names


st.set_page_config(
    page_title="Car Detection",
    page_icon="🚘",
)

# loading the model
with st.spinner(text="Loading Model into the memory..."):
    model, class_names = load_the_pretrained_model()

# setting title of the web app
st.markdown('## Car Image Classification & Localization')

# giving info of the app in the expander
with st.expander("ℹ️ - About this app", expanded=False):
    st.write(
        """
-   Our model is built with 8000+ train images from 196 classes. It will detect the car model with 
    bounding box(localization).
-   Model is trained with MobileNet as base feature extractor and two paths were taken from it one for classification 
    and another for bounding box.
-   Please see the github repository for detailed information about the project 
[car-model-classification-and-localization](https://github.com/aravind-etagi/car-model-classification-and-localization)
-   Linkedin : [Aravind E B](https://www.linkedin.com/in/aravind-eb-38340ab1/)
"""
    )
st.write("\n")

st.write('Either upload your own image or select from the sidebar to get a preconfigured image. '
         'The image you select or upload will be fed through the Deep Neural Network in real-time and the output will'
         ' be displayed to the screen.')

# taking input image either from provided sample images or from uploaded image from user
selected_image = st.sidebar.selectbox('Sample images', os.listdir('./images'))
upload_image = st.file_uploader('Upload an image containing car', accept_multiple_files=False)

if upload_image is not None:
    # conveting images from byte string to numpy array
    uploaded_image = cv2.imdecode(np.frombuffer(upload_image.getvalue(), np.uint8), -1)
    uploaded_image = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)
    msg = 'Making predictions for **uploaded** image'
else:
    # reading the image and converting it into numpy array
    selected_image_link = './images/' + selected_image
    uploaded_image = cv2.imread(selected_image_link)
    uploaded_image = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)
    msg = 'Making predictions for **selected** image from side bar'

# once the predict button is pressed
if st.button('Predict'):
    st.markdown(msg)
    # predicting from preloaded model by calling helper functions
    with st.spinner(text="Predicting..."):
        car_name, df, class_confidence, bbox = predict_image(uploaded_image, IMG_SIZE, model, class_names)
        pred_image = image_with_bbox(uploaded_image, bbox)
        if upload_image is not None:
            uploaded_image_name = upload_image.name
        else:
            uploaded_image_name = selected_image
        h, w, _ = uploaded_image.shape
        code = {'class': car_name,
                'confidence': round(class_confidence, 3),
                'bbox': dict(zip(['x', 'y', 'w', 'h'],list(bbox)))}

    st.write("\n")

    if uploaded_image is not None:
        st.markdown('#### Predicted image')
        st.json(code)                                  # displaying the values as json format
        # displaying the image with bounding box
        st.image(pred_image, use_column_width=True, caption=f'{uploaded_image_name} (height-{h} & width-{w})')


    st.write("\n")

    # displaying top five class predictions with confidence in the tabel format with help of pandas and matplotlib
    st.subheader('Here are the five most likely car models')
    df['Car Name'] = df['Car Name'].apply(
        lambda x: f"""<a href="https://www.google.com/search?q={x.replace(' ', '+')}" 
        target="_blank" 
        rel="noopener noreferrer">{x}</a>""")
    styler = df.style.hide_index().format(subset=['Confidence Level'], decimal='.', precision=3).background_gradient(
        subset=['Confidence Level'])

    st.write(styler.to_html(), unsafe_allow_html=True)
