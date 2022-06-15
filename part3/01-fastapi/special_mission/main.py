from fastapi import FastAPI, UploadFile, File
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any

from datetime import datetime

from special_mission.model import create_trash_label_colormap, label_to_color_image, get_model

import torch, torchvision
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import pandas as pd
import numpy as np
from PIL import Image
import io
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import webcolors

app = FastAPI()

class Product(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    result: Any

@app.get("/")
def hello_world():
    return {"hello": "world"}

@app.post("/inference", description="semantic segmentation inference를 수행합니다.")
async def inference(img_file: UploadFile=File(...),
                    model: Any=Depends(get_model)):
    image_bytes = await img_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    inference_result = inference_segmentor(model, np.array(image))
    res = Product(result=inference_result[0].tolist())
    # print(inference_result[0].tolist())

    return res
