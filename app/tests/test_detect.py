import pytest
import sys
import os

# pytest app import fix
dynamic_path = os.path.abspath('.')
print(dynamic_path)

sys.path.append(dynamic_path)

from func import detect





sample_text = open("tests\sample.txt", "r")
sample_text = sample_text.read()

################################ Test #####################################################

def test_preprocess(raw_text):
    """
    Test to check if the function 'preprocess' is Spliting raw text into list of blocks.
    """
    output = detect.preprocess(sample_text)
    assert output.type == "RGB"

def test_get_bytes_from_image(input_image):
    """
    Test to check if the function 'get_bytes_from_image' is converting the PIL image object to binary image data.
    """
    output = get_bytes_from_image(input_image)
    assert isinstance(output, io.BytesIO)

def test_initialize_models():
    """
    Test to check if all the models are loading correctly.
    """
    model_sample_model = YOLO("./models/sample_model/yolov8n.pt")
    assert model_sample_model is not None

def test_transform_predict_to_df(predictions):
    """
    Test the function 'transform_predict_to_df' which converts the predictions from the YOLO model to a pandas DataFrame.
    It takes in two arguments:
        predictions: A list of dictionaries returned by the YOLO model
        label_names: A list of class labels for the YOLO model
    It returns a DataFrame with columns:
        'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'
    Asserts:
        - The returned object is a DataFrame
        - The columns of the DataFrame are as expected
        - The DataFrame contains at least one object of class 'dog'
    """
    predictions, label_names = predictions
    predict_bbox = transform_predict_to_df(predictions, label_names)
    # Check if the returned object is an instance of pd.DataFrame
    assert isinstance(predict_bbox, pd.DataFrame)
    # Check if the returned DataFrame has the correct columns
    assert set(predict_bbox.columns) == set(['xmin', 'ymin', 'xmax','ymax', 'confidence', 'class', 'name'])
    assert 'dog' in predict_bbox.name.tolist()

def test_get_model_predict(input_image):
    """
    Test to check if the function 'get_model_predict' is returning a DataFrame object with the correct columns and number of rows.
    It also checks if the returned object is an instance of pd.DataFrame
    """
    model_sample_model = YOLO("./models/sample_model/yolov8n.pt")
    predictions = get_model_predict(model_sample_model, input_image)
    # Check if the returned object is an instance of pd.DataFrame
    assert isinstance(predictions, pd.DataFrame)
    # Check if the returned DataFrame has the correct columns
    assert set(predictions.columns) == set(['xmin', 'ymin', 'xmax','ymax', 'confidence', 'class', 'name'])
    # Check if the returned DataFrame has more than one row
    assert len(predictions) > 1

def test_add_bboxs_on_img(input_image, predictions):
    """
    Test to check if the function 'add_bboxs_on_img' is adding bounding boxes on the image and returning the image object.
    """
    predictions, label_names = predictions
    predict_bbox = transform_predict_to_df(predictions, label_names)
    image_with_bbox = add_bboxs_on_img(input_image, predict_bbox)
    assert isinstance(image_with_bbox, Image.Image)