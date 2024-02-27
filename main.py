import cv2
import numpy as np
from paddleocr import PaddleOCR
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

ocr_model = PaddleOCR(use_angle_cls=True, use_gpu=False, lang='en')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "API for Paddle OCR pipeline"}

@app.post("/upload")
async def upload_and_ocr(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray_img, 11, 17, 17)
        blurred_image = cv2.GaussianBlur(bfilter, (5, 5), 0)
        _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = ocr_model.ocr(thresholded_image, cls=False)

        ocr_texts = []
        for line in result:
            for word_info in line:
                ocr_texts.append(word_info[1][0])  # Extract text from word_info tuple
        
        return {"ocr_text": ocr_texts}
    
    except Exception as e:
        return {"error": str(e)}
