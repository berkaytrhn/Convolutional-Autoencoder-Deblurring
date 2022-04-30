from ast import Bytes
from fastapi import FastAPI, UploadFile, File, Form, Request, Response
from typing import Optional
import uvicorn
import sys
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from fastapi.middleware.cors import CORSMiddleware
import sys
sys.path.append("./src/")
from src.predict import *
from fastapi.responses import JSONResponse
import argparse
import json



model = load_sisr_model()

application = FastAPI()


application.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]    
)


@application.get("/test/")
async def test():
    return {"Server is Running!!!"}


@application.post("/upload/", response_class=JSONResponse)
async def upload(request: Request):
    data= np.array(await request.json())
    # remove alpha channel from input image
    data=data.reshape((IMAGE_SIZE, IMAGE_SIZE, 4))[:,:,:3].astype(np.float32)

    data=cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    # normalization
    data/=255.0

    # model prediction
    result=predict(data, model)

    # convert to bgra for opencv.js
    result=cv2.cvtColor(result, cv2.COLOR_RGB2BGRA)

    # convert to json for sending
    result=json.dumps(result.tolist())
    return result
 




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ip", "-i", required=True)
    parser.add_argument("--port", "-p", required=True, type=int)

    args = parser.parse_args()


    uvicorn.run("app:application", host=args.ip, port=args.port, reload=True)
