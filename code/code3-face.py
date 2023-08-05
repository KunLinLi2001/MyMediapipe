# 导入必要的库
import cv2
import mediapipe as mp
import time

# 计算两点间距离的函数
def findDis(pts1, pts2):
    return ((pts2[0] - pts1[0]) ** 2 + (pts2[1] - pts1[1]) ** 2) ** 0.5

# 获取摄像头
cap = cv2.VideoCapture(0)
# 记录上一个帧的时间
pTime = 0

# 需要检测的人脸特征点id
id_list = [23, 159, 130, 243, 62, 292, 12, 15]

# 创建画布
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

# 循环读取视频流
while True:
    success, img = cap.read()
    # 转为RGB格式
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 处理人脸特征点
    results = faceMesh.process(imgRGB)
    # 如果检测到人脸特征点
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            # 绘制人脸特征点
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            mp_data = []
            # 遍历每个特征点
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                if id in id_list:  # 左眼[22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]:
                    # 如果是需要检测的眼睛或嘴巴特征点，记录到mp_data中
                    mp_data.append([x, y])
                    cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
            # 根据特征点计算眼睛和嘴巴的长度
            eye_length_1 = findDis(mp_data[0], mp_data[1])
            eye_length_2 = findDis(mp_data[2], mp_data[3])
            mouth_length_2 = findDis(mp_data[4], mp_data[5])
            mouth_length_1 = findDis(mp_data[6], mp_data[7])
            # 判断嘴巴是否闭合
            if ((mouth_length_1 / mouth_length_2) < (98 / 18)):
                cv2.putText(img, "mouth close", (400, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            else:
                cv2.putText(img, "mouth open", (400, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            # 判断眼睛是否闭合
            if (eye_length_2 / eye_length_1) > 18:
                cv2.putText(img, "eye open", (400, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
            else:
                cv2.putText(img, "eye close", (400, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    # 计算并绘制帧率
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    # 按q键退出循环
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.imwrite("6.jpg", img)
        break
# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
