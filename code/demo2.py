import mediapipe as mp
import cv2
import numpy as np
import time


# 定义一个函数，计算两个点的距离
def findDis(pts1,pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5
# 创建手势检测模型
mpHands = mp.solutions.hands  # 检测人的手
hand_mode = mpHands.Hands(max_num_hands=2,min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
# static_image_mode：默认为False，如果设置为false, 就是把输入看作一个视频流，在检测到手之后对手加了一个目标跟踪(目标检测+跟踪)，
# 无需调用另一次检测，直到失去对任何手的跟踪为止。如果设置为True，则手部检测将在每个输入图像上运行(目标检测)，非常适合处理一批静态的，
# 可能不相关的图像。(如果检测的是图片就要设置成True)
# 检测手的模式参数设置，max_num_hands:可以检测到的手的数量最大值，默认是2
# min_detection_confidence: 手部检测的最小置信度值，大于这个数值被认为是成功的检测，
# min_tracking_confidence：目标踪模型的最小置信度值，大于这个数值将被视为已成功跟踪的手部，如果static_image_mode设置为true，则忽略此操作。
mpDraw = mp.solutions.drawing_utils  # 绘图

cap = cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    img = cv2.flip(img,1)
    results = hand_mode.process(img)# 将图片导入模型，获取20个点的坐标进行分析
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            point4_8 = []
            for id,lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                cv2.circle(img,(cx,cy),10,(255,0,0),-1)
                if id in [4,8]:# 获取点4，8的坐标
                    point4_8.append([cx,cy])
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            # 求点4，8的坐标，进行可视化展示
            cv2.line(img,(point4_8[0][0],point4_8[0][1]),(point4_8[1][0],point4_8[1][1]),(0,0,255),5)
            distance = round(findDis((point4_8[0][0],point4_8[0][1]),(point4_8[1][0],point4_8[1][1])),2)
            cv2.putText(img,"distance:{}".format(distance),(50,50),cv2.FONT_HERSHEY_PLAIN, 3,(0,0,255),3)
            cv2.rectangle(img,(20,250),(20+10,250-int(distance)),(255,0,255),20)
    cv2.imshow("img",img)
    if cv2.waitKey(1)&0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
