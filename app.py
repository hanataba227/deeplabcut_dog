import os
import cv2
import glob
import numpy as np
import pandas as pd
import deeplabcut
import shutil
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# FastAPI 앱 생성
app = FastAPI()

# 정적 파일 폴더 설정
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/results", StaticFiles(directory="results"), name="results")

# Jinja2 템플릿 설정
templates = Jinja2Templates(directory="templates")

# 파일 저장 폴더 설정
UPLOAD_FOLDER = "C:/project/dog/uploads"
RESULT_FOLDER = "C:/project/dog/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# DeepLabCut 프로젝트 설정
project_name = "dog_pose_project"
researcher_name = "user"

# ✅ 사전 학습된 모델로 프로젝트 생성
config_path_tuple = deeplabcut.create_pretrained_project(
    project=project_name, experimenter=researcher_name, videos=[], model="full_dog"
)
config_path = config_path_tuple[0]
print(f"✅ DeepLabCut 프로젝트 생성 완료: {config_path}")

def process_image(image_path):
    """ 📌 이미지를 512x512 크기로 리사이징하여 저장 (컬러 유지) """
    resized_path = image_path.replace(".jpg", "_resized.jpg")
    image = cv2.imread(image_path)  # 원본 컬러 이미지 로드
    if image is None:
        raise ValueError(f"❌ 이미지 로드 실패: {image_path}")

    original_height, original_width = image.shape[:2]
    
    # ✅ 컬러 이미지 그대로 리사이징
    resized_image = cv2.resize(image, (512, 512))
    cv2.imwrite(resized_path, resized_image)  # ✅ 컬러 이미지 저장

    return resized_path, resized_image, original_width, original_height

def extract_keypoints(image_path):
    """ 📌 DeepLabCut을 이용해 Keypoint를 추출하고, 결과 CSV 경로 반환 """
    resized_path, resized_image, original_width, original_height = process_image(image_path)

    deeplabcut.analyze_videos(config_path, [resized_path], save_as_csv=True)

    # ✅ CSV 파일 검색
    csv_files = glob.glob(f"{os.path.dirname(resized_path)}/*DLC*.csv")
    
    if len(csv_files) == 0:
        print("❌ CSV 파일을 찾을 수 없습니다. 경로 확인:", os.path.dirname(resized_path))
        return None, None, None, None  # 🔴 값이 없을 경우 `None`을 반환하여 방지

    return csv_files[0], resized_image, original_width, original_height  # 🔥 올바른 값 반환

def calculate_body_measurements(csv_path, original_width, original_height):
    """ 📌 CSV에서 Keypoint를 읽고, 리사이징된 이미지 크기에 맞게 좌표 변환 """
    if csv_path is None:
        return None, None  # 🔴 CSV가 없으면 실행 방지

    keypoints_df = pd.read_csv(csv_path, index_col=0, header=[1, 2])

    keypoints = ["Nose", "Throat", "Withers", "TailSet",
                 "L_F_Paw", "R_F_Paw", "L_F_Wrist", "R_F_Wrist", "L_F_Elbow", "R_F_Elbow",
                 "L_B_Paw", "R_B_Paw", "L_B_Hock", "R_B_Hock", "L_B_Stiffle", "R_B_Stiffle"]

    keypoint_positions = {}
    for keypoint in keypoints:
        if (keypoint, 'x') in keypoints_df.columns and (keypoint, 'y') in keypoints_df.columns:
            x, y = keypoints_df[(keypoint, 'x')].values[0], keypoints_df[(keypoint, 'y')].values[0]

            # ✅ 올바른 스케일 적용
            x_resized = int((x / 512) * original_width)
            y_resized = int((y / 512) * original_height)

            keypoint_positions[keypoint] = (x_resized, y_resized)

    return keypoint_positions

def visualize_keypoints(image, image_path, keypoint_positions):
    """ 📌 리사이징된 이미지에 Keypoint를 시각화하고 저장 """
    if keypoint_positions is None:
        return None  # 🔴 Keypoint가 없으면 실행 방지

    result_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path).replace(".jpg", "_result.jpg"))

    # ✅ 주요 부위 연결 (선 그리기)
    lines = [
        ("Throat", "Withers"), ("Withers", "TailSet"),
        ("Withers", "R_F_Elbow"), ("R_F_Elbow", "R_F_Wrist")
    ]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    for (pt1, pt2), color in zip(lines, colors):
        if pt1 in keypoint_positions and pt2 in keypoint_positions:
            cv2.line(image, keypoint_positions[pt1], keypoint_positions[pt2], color, 3)

    # ✅ Keypoint 원 표시
    for keypoint, (x, y) in keypoint_positions.items():
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(image, keypoint, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(result_path, image)
    print(f"✅ 결과 이미지 저장 완료: {result_path}")
    return result_path

@app.get("/", response_class=HTMLResponse) 
async def home(request: Request): 
    """ 웹 인터페이스 렌더링 """ 
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_image(request: Request, file: UploadFile = File(...)):
    """ 📌 이미지 업로드 및 분석 실행 """
    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    csv_path, resized_image, original_width, original_height = extract_keypoints(file_path)
    keypoint_positions = calculate_body_measurements(csv_path, original_width, original_height)
    result_path = visualize_keypoints(resized_image, file_path, keypoint_positions)

    return JSONResponse({"image_url": f"/uploads/{filename}", "result_image": f"/results/{os.path.basename(result_path)}"})
