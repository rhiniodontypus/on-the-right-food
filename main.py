import streamlit as st

import os
import cv2
import random
import string

import pandas as pd
import plotly.express as px

from PIL import Image

from pycocotools.coco import COCO

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# import configuration file settings.py
from config import settings

st.set_page_config(page_title="On the right food", layout="wide")

# defining the paths
OUTPUT_DIR = settings.OUTPUT_DIR
path_train_subset_annotations = settings.PATH_TRAIN_SUBSET_ANNOTATIONS
path_train_subset_images = settings.PATH_TRAIN_SUBSET_IMAGES
trained_model = settings.TRAINED_MODEL
cached_folder = settings.CACHED_FOLDER

# creating a random string for coco dataset name
@st.cache_resource
def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))
rdm_dataset_name = randomword(10)

# registering the coco dataset
@st.cache_resource
def load_coco_dataset():
    register_coco_instances(
        rdm_dataset_name,
        {},
        path_train_subset_annotations,
        path_train_subset_images
    )
    # Loading the dataset
    metadata = MetadataCatalog.get(rdm_dataset_name)
    dataset_dicts = DatasetCatalog.get(rdm_dataset_name)
    return metadata, dataset_dicts
metadata, dataset_dicts = load_coco_dataset()

# counting the total number of food classes in the annotation file
@st.cache_resource
def food_classes():
    #coco=COCO(config.PATH_TRAIN_SUBSET_ANNOTATIONS)
    coco=COCO(path_train_subset_annotations)
    number_classes = len(coco.getCatIds())
    return number_classes
number_classes = food_classes()


# Creating a detectron2 config and a detectron2 'DefaultPredictor' to run inference on the image.
# setting default threshold
def predictor():
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = number_classes
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, trained_model)
    cfg.DATASETS.TRAIN = (rdm_dataset_name)
    cfg.DATASETS.TEST = ()
    return cfg

# setting threshold with the slider
def predictor_threshold(threshold_to_filter):
    cfg_pred_slider = get_cfg()
    cfg_pred_slider.MODEL.DEVICE = "cpu"
    cfg_pred_slider.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg_pred_slider.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold_to_filter/100
    cfg_pred_slider.MODEL.ROI_HEADS.NUM_CLASSES = number_classes
    cfg_pred_slider.MODEL.WEIGHTS = os.path.join(cfg_pred_slider.OUTPUT_DIR, trained_model)
    cfg_pred_slider.DATASETS.TRAIN = (rdm_dataset_name)
    cfg_pred_slider.DATASETS.TEST = ()
    return cfg_pred_slider

###########################################
# Web app elements
###########################################

# creating the sidebar
with st.sidebar:
    st.subheader('**Prediction probability (in %):**')
    threshold_to_filter = st.slider('', 0, 100, 70)

# setting header for website
st.header('Take a picture of your :blue[food] :knife_fork_plate:!')

col1, col2, col3 = st.columns(3)
with col1:
    st.header('See what you eat!')
with col3:
    # Displaying the logo
    st.image('./images/on_the_right_food_logo_dark.png', width=250)

# Displaying a file uploader widget
uploaded_file = st.file_uploader("Choose an image file of type jpg/jpeg: ")

# If the user uploaded a file
if uploaded_file is not None:  

    # opening uploaded file and caches a copy
    uploaded_file = Image.open(uploaded_file)
    #uploaded_file_new = uploaded_file.save('cv2_img.jpg')
    uploaded_file_new = uploaded_file.save(cached_folder + 'cv2_img_cache.jpg')
    im = cv2.imread(cached_folder + 'cv2_img_cache.jpg')
    
    # getting image details
    width = im.shape[0]
    height = im.shape[1]
    channels = im.shape[2]

    # showing original image
    st.header("Original")
    st.image(uploaded_file, width=300)
    # printing image details
    st.write('Resolution: ', 
             str(width),'x',
             str(height),
             ',','Channels: ',
             str(channels))


    ###########################################
    # creating images with prediction overlays
    ###########################################
   
    def image_overlay():
        # getting the predictions both for all classes and for the prediction slider
        predictor_all_classes = DefaultPredictor(predictor())
        predictor_slider = DefaultPredictor(predictor_threshold(threshold_to_filter))
        
        # creating image with food classes by threshold
        outputs_slider = predictor_slider(im)
        # creating image with all food classes
        outputs_all_classes = predictor_all_classes(im)
        v_slider = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=1,
                       instance_mode=ColorMode.IMAGE_BW     # removing the colors of unsegmented pixels.          
            )        
        v_all = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=1,
                       instance_mode=ColorMode.IMAGE_BW                      
            )

        return (v_slider.draw_instance_predictions(outputs_slider["instances"].to("cpu")),
                outputs_slider, 
                v_all.draw_instance_predictions(outputs_all_classes["instances"].to("cpu")),
                outputs_all_classes)

    out_slider, outputs_slider, out_all, outputs_all_classes = image_overlay()


    ########################################################
    # extracting classes, segmentations and prediction scores for plotting
    ########################################################

    # extracting all predicted food classes
    def extract_food_classes():
        # extracting all predicted food classes
        pred_classes_all = outputs_all_classes['instances'].pred_classes.tolist()
        class_names_all = metadata.thing_classes
        pred_class_names_all = list(map(lambda x: class_names_all[x], pred_classes_all))
        pred_class_score_all = outputs_all_classes['instances'].scores.tolist()

        # extracting the predicted food classes selected with slider threshold
        pred_classes_slider = outputs_slider['instances'].pred_classes.tolist()
        class_names_slider = metadata.thing_classes
        pred_class_names_slider = list(map(lambda x: class_names_slider[x], pred_classes_slider))
        pred_class_score_slider = outputs_slider['instances'].scores.tolist()
        
        # calculating the area of each segmentation
        area = [sum(sum(list_x)) for list_x in outputs_slider['instances'].pred_masks]
        total_area = 0
        area_list = []
        for area_segs in area:
            area_segs1 = (str(area_segs)).replace('(','').replace(')','').replace('tensor','')
            area_segs = (int(str(area_segs1)))
            total_area += area_segs
            area_list.append(area_segs)

        return (
                pred_class_names_all,
                pred_class_score_all,
                pred_class_names_slider,
                pred_class_score_slider,
                area_list
               )

    (
     pred_class_names_all, pred_class_score_all,
     pred_class_names_slider, pred_class_score_slider,
     area_list
    ) = extract_food_classes()


    ########################################################
    # creating dataframes from prediction scores for plotting
    ########################################################
    
    def prediction_scores_dataframe():
        # creating dataframe for the prediction scores for all classes
        res_all_scores = {'Food category': [] , 'Prediction Score': []}
        values_all_scores = [
                        pred_class_names_all,
                        (pd.Series(pred_class_score_all) * 100)
        ]
        for i, key in enumerate(res_all_scores.keys()):
            res_all_scores[key] = values_all_scores[i]
        df_all_scores = pd.DataFrame.from_dict(res_all_scores)
        # getting only the first (highest) entry of a food category
        df_all_scores_first = df_all_scores.groupby('Food category').first().sort_values(by=['Prediction Score']).reset_index()

        # creating dataframe for the prediction scores of the selected classes by threshold
        res_slide_scores = {'Food category': [] , 'Prediction Score': []}
        values_slide_cores = [
                        pred_class_names_slider, 
                        (pd.Series(pred_class_score_slider) * 100)
        ]
        for i, key in enumerate(res_slide_scores.keys()):
            res_slide_scores[key] = values_slide_cores[i]
        df_slide_scores = pd.DataFrame.from_dict(res_slide_scores)
        # getting only the first (highest) entry of a food category
        df_slide_scores_first = df_slide_scores.groupby('Food category').first().sort_values(by=['Prediction Score']).reset_index()

        # creating dataframe with food categories and segmentation area size
        res_slider_area = {'Food category': [] , 'Area': []}
        values_slider_area = [pred_class_names_slider, area_list]
        for i, key in enumerate(res_slider_area.keys()):
            res_slider_area[key] = values_slider_area[i]
        df_slider_area = pd.DataFrame.from_dict(res_slider_area)

        return df_all_scores_first, df_slide_scores_first, df_slider_area
    
    (
     df_all_scores_first,
     df_slide_scores_first,
     df_slider_area
    ) = prediction_scores_dataframe()


    ###########################################
    # creating tabs
    ########################################### 

    tab1, tab2, = st.tabs(["Food categories", "Food quantity"])  

    with tab1:
        ###########################################
        # creating columns for the first tab
        ########################################### 
        col1, col2 = st.columns(2)

        with col1:
            ###########################################
            # showing image and plot with all food classes predictions
            ########################################### 
            st.header("All predictions")
            st.image(out_all.get_image()[:, :, ::-1])

            # creating plot for food classes by default prediction value
            def plot_classes_default_prediction_score():
                df_all_scores_first['Prediction Score'] = df_all_scores_first['Prediction Score'].round(0)
                fig_all_scores = px.bar(
                    #df_all_scores,
                    df_all_scores_first,
                    y="Food category",
                    x="Prediction Score",
                    text="Prediction Score",
                    title="Prediction score for all identified classes (>10%)",
                    labels={
                                "Food category": "Food category",
                                "Prediction Score": "Prediction Score in %"
                            },
                    orientation='h',
                )
                return fig_all_scores

            st.plotly_chart(plot_classes_default_prediction_score(), theme=None, use_container_width=True)

        with col2:
            ###########################################
            # showing image and plot with food classes predictions filtered by threshold
            ###########################################

            st.header("Predictions by threshold")
            st.image(out_slider.get_image()[:, :, ::-1])

            # creating plot for food classes by threshold preciction value
            def plot_classes_threshold_prediction_score():
                df_slide_scores_first['Prediction Score'] = df_slide_scores_first['Prediction Score'].round(0)
                fig_pred_score_thresh = px.bar(
                    df_slide_scores_first,
                    #get_df_slider(),   
                    y="Food category",
                    x="Prediction Score",
                    text="Prediction Score",
                    title="Prediction score above the selected threshold",
                    labels={
                                "Food category": "Food category",
                                "Prediction Score": "Prediction Score in %"
                            },
                    orientation='h',
                )
                return fig_pred_score_thresh

            st.plotly_chart(plot_classes_threshold_prediction_score(), theme=None, use_container_width=True)

    ###########################################
    # Showing size of segmentation areas in second tab
    ###########################################
    
    with tab2:
        st.header('Food quantity determination')
        st.image(out_slider.get_image()[:, :, ::-1], width=500)

        # creating plotly express figure for pixel count (segmentation area)
        def plot_food_segment_areas():
            fig_seg_area = px.bar(
                df_slider_area.sort_values(by=['Area']), 
                y="Food category",
                x="Area",
                title="Relative area of food segments",
                text = "Area",
                labels={
                            "Food category": "Food category",
                            "Area": "Relative area of food segments in pixels"
                        },
                orientation='h',
            )
            return fig_seg_area


        col_fq1, col_fq2 = st.columns((2, 1))
        with col_fq1:
            st.plotly_chart(plot_food_segment_areas(), theme="streamlit", use_container_width=True)