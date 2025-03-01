import deeplabcut
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob

### 📌 1) 프로젝트 설정 ###
# ✅ 분석할 강아지 원본 이미지 경로
image_path = "C:/project/dog/main/dataset/images/train/B_10_MAL_IF_20221125_10_101177_07.jpg"
resized_image_path = image_path.replace(".jpg", "_resized.jpg")  # 리사이징된 이미지 저장 경로

# ✅ 원본 이미지를 리사이징 후 저장
image = cv2.imread(image_path)
image_resized = cv2.resize(image, (512, 512))
cv2.imwrite(resized_image_path, image_resized)

# ✅ 프로젝트 및 연구자 이름 설정
project_name = "dog_pose_project"
researcher_name = "user"

# ✅ DeepLabCut 프로젝트 생성 (사전 학습된 모델 사용)
config_path_tuple = deeplabcut.create_pretrained_project(
    project=project_name,
    experimenter=researcher_name,
    videos=[resized_image_path],  # 🔹 리사이징된 이미지 사용
    model="full_dog"
)
config_path = config_path_tuple[0]  # ✅ config.yaml 경로

print(f"✅ DeepLabCut 프로젝트 생성 완료: {config_path}")

### 📌 2) 강아지 Keypoint 추출 ###
# ✅ Keypoint 분석 수행
deeplabcut.analyze_videos(config_path, [resized_image_path], save_as_csv=True)

# ✅ CSV 파일 자동 탐색 (프로젝트 날짜가 다를 경우 대비)
csv_pattern = resized_image_path.replace(".jpg", "DLC_resnet50_dog_pose_project*shuffle1_75000.csv")
csv_files = glob.glob(csv_pattern)

if len(csv_files) == 0:
    raise FileNotFoundError(f"🔴 CSV 파일을 찾을 수 없습니다: {csv_pattern}")
else:
    csv_path = csv_files[0]  # ✅ 가장 최신 파일 선택

print(f"✅ 분석된 CSV 파일: {csv_path}")

# ✅ CSV 파일 로드 (MultiIndex 처리)
keypoints_df = pd.read_csv(csv_path, index_col=0, header=[1, 2])

# ✅ 가져올 주요 Keypoint 목록
keypoints = ["Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear", "Throat", "Withers", "TailSet",
             "L_F_Paw", "R_F_Paw", "L_F_Wrist", "R_F_Wrist", "L_F_Elbow", "R_F_Elbow",
             "L_B_Paw", "R_B_Paw", "L_B_Hock", "R_B_Hock", "L_B_Stiffle", "R_B_Stiffle"]

# ✅ Keypoint 좌표 가져오기
keypoint_positions = {}
for keypoint in keypoints:
    x, y = keypoints_df[(keypoint, 'x')].values[0], keypoints_df[(keypoint, 'y')].values[0]
    keypoint_positions[keypoint] = (int(x), int(y))

### 📌 3) Keypoint를 활용한 신체 치수 측정 ###
def euclidean_distance(pt1, pt2):
    """ 두 Keypoint 사이의 거리 계산 (픽셀 단위) """
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

# ✅ 신체 측정값 계산 (픽셀 단위)
neck_size_px = euclidean_distance(keypoint_positions["Throat"], keypoint_positions["Withers"])  # 목둘레
chest_size_px = euclidean_distance(keypoint_positions["Withers"], keypoint_positions["R_F_Elbow"])  # 가슴둘레
back_length_px = euclidean_distance(keypoint_positions["Withers"], keypoint_positions["TailSet"])  # 등길이
leg_length_px = euclidean_distance(keypoint_positions["R_F_Elbow"], keypoint_positions["R_F_Wrist"])  # 다리길이

# ✅ 신체 측정 결과 출력
print("\n📏 신체 치수 (픽셀 단위)")
print(f"목둘레: {neck_size_px:.2f} px")
print(f"가슴둘레: {chest_size_px:.2f} px")
print(f"등길이: {back_length_px:.2f} px")
print(f"다리길이: {leg_length_px:.2f} px")

### 📌 4) 이미지에 Keypoint 시각화 ###
# ✅ 리사이징된 이미지 불러오기 (결과 시각화용)
image = cv2.imread(resized_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ✅ 주요 부위 연결 (선으로 표시할 부위 정의)
lines = {
    "neck": ["Throat", "Withers"],
    "chest": ["Withers", "R_F_Elbow"],
    "back": ["Withers", "TailSet"],
    "leg": ["R_F_Elbow", "R_F_Wrist"]
}

# ✅ 선 그리기 (각 부위별 색상 설정)
colors = {
    "neck": (255, 0, 0),    # 🔵 파란색
    "chest": (0, 255, 0),   # 🟢 초록색
    "back": (0, 0, 255),    # 🔴 빨간색
    "leg": (255, 255, 0)    # 🟡 노란색
}

for part, points in lines.items():
    for i in range(len(points) - 1):
        pt1 = keypoint_positions[points[i]]
        pt2 = keypoint_positions[points[i + 1]]
        cv2.line(image, pt1, pt2, colors[part], 3)

# ✅ Keypoint 점 찍기 (연결된 부위 외 나머지)
for keypoint, (x, y) in keypoint_positions.items():
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # 초록색 점
    cv2.putText(image, keypoint, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

### 📌 5) 결과 시각화 ###
plt.figure(figsize=(10, 6))
plt.imshow(image)
plt.axis("off")
plt.title("📏 Keypoint Detection & 신체 치수 시각화")
plt.show()
