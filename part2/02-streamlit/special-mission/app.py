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


def create_trash_label_colormap(class_colormap):
    """Creates a label colormap used in Trash segmentation.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((11, 3), dtype=np.uint8)
    for inex, (_, r, g, b) in enumerate(class_colormap.values):
        colormap[inex] = [r, g, b]
    
    return colormap


def label_to_color_image(label, class_colormap):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
                is the color indexed by the corresponding element in the input label
                to the trash color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
              map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_trash_label_colormap(class_colormap)

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


# def hash_func(obj):
#     return [obj.detach().cpu().numpy(), obj.grad]

# @st.cache(hash_funcs={"torch.nn.parameter.Parameter": hash_func})
# def load_model(config_file, checkpoint_file, device):
#     model = init_segmentor(config_file, checkpoint_file, device=device)
#     return model


def main():
    st.header("Semantic Segmentation for Recycling ♻️")
    st.write("**Name** : 유승리_T3129 (CV-10) / **Backbone** : Swin-L / **Decoder** : UPerNet / **Etc** : no fold & pseudo labeling")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config_file = './assets/0511-upernet_swin_l_full_pl.py'
    checkpoint_file = './assets/epoch_46.pth'
    class_colormap = pd.read_csv("./assets/class_dict.csv")

    if 'model' not in st.session_state:
        st.session_state.model = None
        with st.spinner('Loading the model...'):
            st.session_state.model = init_segmentor(config_file, checkpoint_file, device=device)
    # with st.spinner('Loading the model...'):
    #     model = load_model(config_file, checkpoint_file, device=device) 

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg","png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

        st.subheader("Uploaded Image")
        st.image(image, caption='You can check your image.')
        st.subheader("Inference Result")

        with st.spinner('Inferencing...'):
            result = inference_segmentor(st.session_state.model, np.array(image))

            # plot
            category_and_rgb = [[category, (r,g,b)] for idx, (category, r, g, b) in enumerate(class_colormap.values)]
            legend_elements = [Patch(facecolor=webcolors.rgb_to_hex(rgb), 
                                edgecolor=webcolors.rgb_to_hex(rgb), 
                                label=category) for category, rgb in category_and_rgb]
            res_img = show_result_pyplot(st.session_state.model, np.array(image), result)
            fig, ax = plt.subplots(nrows=2, figsize=(10, 20))
            ax[0].set_title("Prediction")
            ax[0].imshow(res_img)
            ax[1].set_title("Pred Mask")
            ax[1].imshow(label_to_color_image(result[0], class_colormap))
            ax[1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
            st.pyplot(fig)
            # st.image(label_to_color_image(result[0], class_colormap), caption='Inference Result')

main()
