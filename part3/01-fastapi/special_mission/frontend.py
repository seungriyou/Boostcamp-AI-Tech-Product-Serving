import streamlit as st
import torch, torchvision
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import pandas as pd
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import webcolors
import os
import requests
from model import label_to_color_image


class_colormap = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/class_dict.csv'))


def main():
    st.header("Semantic Segmentation for Recycling ♻️")
    st.write("**Name** : 유승리_T3129 (CV-10) / **Backbone** : Swin-L / **Decoder** : UPerNet / **Etc** : no fold & pseudo labeling")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg","png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        img_file = [('img_file', (uploaded_file.name, image_bytes,
                       uploaded_file.type))]
        image = Image.open(io.BytesIO(image_bytes))

        st.subheader("Uploaded Image")
        st.image(image, caption='You can check your image.')
        st.subheader("Inference Result")

        with st.spinner('Inferencing...'):
            response = requests.post("http://118.67.130.53:30001/inference", files=img_file)
            
            category_and_rgb = [[category, (r,g,b)] for idx, (category, r, g, b) in enumerate(class_colormap.values)]
            legend_elements = [Patch(facecolor=webcolors.rgb_to_hex(rgb), 
                                edgecolor=webcolors.rgb_to_hex(rgb), 
                                label=category) for category, rgb in category_and_rgb]
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_title("Pred Mask")
            ax.imshow(label_to_color_image(np.array(response.json()["result"]), class_colormap))
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
            st.pyplot(fig)


main()
