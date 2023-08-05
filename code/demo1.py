import cv2
import mediapipe as mp
import time

# 定义一个函数，计算两个点的距离
def findDis(pts1,pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5

cap = cv2.VideoCapture(0)  # 打开摄像头：0=内置摄像头（笔记本）   1=USB摄像头-1   2=USB摄像头-2

mpHands = mp.solutions.hands  # 定义并引用mediapipe中的hands模块
hands = mpHands.Hands()  # 初始化hands模块
mpDraw = mp.solutions.drawing_utils  # 定义并引用mediapipe中的绘图工具

pTime = 0  # 上一帧的时间
cTime = 0  # 本帧的时间

while True:
    success, img = cap.read()  # 读取摄像头捕获的一帧图像
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2图像初始化

    results = hands.process(imgRGB)  # 处理手部特征点

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            point4_8 = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape  # 获取图像的高、宽、通道数
                cx, cy = int(lm.x * w), int(lm.y * h)  # 计算出特征点在图像中的具体位置
                print(id, cx, cy)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)  # 在特征点位置添加圆形标记
                if id in [4,8]:# 获取点4，8的坐标(食指和大拇指）
                    point4_8.append([cx,cy])

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  # 绘制手部特征点和连线
            # 求点4，8的坐标，进行可视化展示
            cv2.line(img,(point4_8[0][0],point4_8[0][1]),(point4_8[1][0],point4_8[1][1]),(0,0,255),5)
            distance = round(findDis((point4_8[0][0],point4_8[0][1]),(point4_8[1][0],point4_8[1][1])),2)
            cv2.putText(img,"distance:{}".format(distance),(100,100),cv2.FONT_HERSHEY_PLAIN, 3,(0,0,255),3)
            cv2.rectangle(img,(20,250),(20+10,250-int(distance)),(255,0,255),20)

    cTime = time.time()  # 获取当前时间
    fps = 1 / (cTime - pTime)  # 计算出当前帧率
    pTime = cTime  # 将本帧时间设为上一帧时间

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  # 在图像上添加FPS信息

    cv2.imshow("HandsImage", img)  # 显示处理后的图像
    cv2.waitKey(1)  # 等待按键或关闭窗口
