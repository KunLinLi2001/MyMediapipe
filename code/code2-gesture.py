# 导入必要的库
import cv2
import mediapipe as mp
import time

# 实例化人体姿态检测模型和绘图工具
mpPose = mp.solutions.pose  # 检测人的姿态
pose_mode = mpPose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5)  # 姿态检测模式参数设置
mpDraw = mp.solutions.drawing_utils  # 绘图

# 打开摄像头
cap = cv2.VideoCapture(0)
flag = 0
i = 0

# 循环读取摄像头，实时检测姿态
while True:
    success,img = cap.read()  # 读取摄像头
    img = cv2.flip(img,1)  # 翻转图像
    results = pose_mode.process(img)  # 姿态检测
    if results.pose_landmarks:  # 如果检测到了姿态关键点
        point23_25 = []  # 记录左右足踝的位置
        for id,lm in enumerate(results.pose_landmarks.landmark):  # 遍历所有关键点
            h,w,c = img.shape
            cx,cy = int(lm.x*w),int(lm.y*h)  # 坐标归一化
            cv2.circle(img,(cx,cy),10,(255,0,0),-1)  # 绘制关键点
            if id in [23,25]:  # 如果是左右足踝的关键点
                point23_25.append([cx,cy])  # 记录其位置
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)  # 绘制关键点连接线
        cv2.line(img,(point23_25[0][0],point23_25[0][1]),(point23_25[1][0],point23_25[1][1]),(0,0,255),5)  # 在足踝处绘制连线
        if point23_25[0][1]>point23_25[1][1]:  # 如果左腿抬高
            if flag == 1:  # 如果之前是下蹲状态
                i += 1  # 记录抬腿次数
                flag = 0  # 标记为上升状态
                cv2.putText(img,"Leg up--{}".format(i),(10,50),cv2.FONT_HERSHEY_PLAIN, 3,(0,0,255),3)  # 在图像上显示抬腿次数
        else:  # 如果左腿下降
            flag = 1  # 标记为下降状态
            cv2.putText(img,"Leg down--{}".format(i),(10,450),cv2.FONT_HERSHEY_PLAIN, 3,(0,0,255),3)  # 在图像上显示下蹲次数
    cv2.imshow("img",img)  # 显示图像
    if cv2.waitKey(1)&0xFF == ord("q"):  # 如果按下q键，退出循环
        break

# 释放摄像头资源，关闭窗口
cap.release()
cv2.destroyAllWindows()
