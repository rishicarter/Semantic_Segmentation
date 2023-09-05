from fastai.vision.all import *
import streamlit as st
from PIL import Image

color_map={
    0:(72,98,116),
    1:(72,136,39),
    2:(201,47,40),
    3:(171,124,122),
    4:(171,124,122)
}

def label_fn(fn):
    pass


def load_segimg_from_model(picture):
    # learn=load_learner(f'models/{model}')
    # def label_fn(fn): 
    #     return path/'semantics'/f'{fn.stem}{fn.suffix}'
    tfms = [Resize(224), IntToFloatTensor(),Normalize()]
    learn0 = load_learner(Path('./models/model_3.pkl'))
    learn0.dls.transform = tfms
    
    pred_mask,_,_=learn0.predict(picture)
    h, w = pred_mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_code, color in color_map.items():
        mask_pixels = (pred_mask == class_code)
        colored_mask[mask_pixels] = color

    # Create a PIL Image from the colored mask
    colored_mask_image = Image.fromarray(colored_mask)
    
    return colored_mask_image



st.title("Crop-Weed Segmentation")

st.header('Segmentation Example')
st.text('Color map - \n Soil - Blue\n Crop - Green \n Weed - Red \n Partial Crop/Weed - Light Red')

col_img,col_seg=st.columns(2)
col_img.text("Example Image")
col_img.image(Image.open(Path('./images/img.png')))
col_seg.text("Segmented Image")
col_seg.image(Image.open(Path('./images/seg.png')))
st.text('\n\n')
picture = st.camera_input("Test!")

if picture:
    col1,col2=st.columns(2)
    col1.subheader("Clicked Image")
    col2.subheader("Segmented Image")
    col3,col4=st.columns(2)
    col3.image(picture)
    
    colored_mask_image=load_segimg_from_model(PILImage.create(picture))
    
    col4.image(colored_mask_image)
    
    

