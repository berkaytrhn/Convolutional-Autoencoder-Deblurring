from ast import Bytes
from fastapi import FastAPI, UploadFile, File, Form, Request
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
from src.utils import *
from src.predict import *
from fastapi.responses import JSONResponse
import argparse



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


@application.post("/upload/")
async def upload(request: Request):
    data= np.array(await request.json())
    # remove alpha channel from input image
    data=data.reshape((IMAGE_SIZE, IMAGE_SIZE, 4))[:,:,:3].astype(np.float32)

    data=cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    # normalization
    data/=255.0

    cv2.imshow("test_initial ", data)

    predict(data, model)


@application.post("/upload_img/", response_class=JSONResponse)
async def upload_img(file: UploadFile = File(...)):
    # get and convert image to numpy array
    data=await file.read()
    print(data)
    img = Image.open(BytesIO(data))
    # convert PIL image to numpy array and 'BGR' to 'RGB'
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
    cv2.imshow("received image ",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """
    try:
        pred, prob, coords = predict(img, model)
        return {
            "Predicted":  str(pred),
            "Probability": str(prob),
            "x_min": str(coords[0]),
            "y_min": str(coords[1]),
            "x_max": str(coords[2]),
            "y_max": str(coords[3])
        }
    except: 
        # face detection failed!!
        return {"No Faces, Try Again!!"}
    """
    

    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ip", "-i", required=True)
    parser.add_argument("--port", "-p", required=True, type=int)

    args = parser.parse_args()


    uvicorn.run("app:application", host=args.ip, port=args.port, reload=True)
