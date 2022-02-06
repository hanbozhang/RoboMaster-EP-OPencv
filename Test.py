import cv2
import numpy as np

#载入测试视频
video=cv2.VideoCapture('Test.mp4')
#获取视频帧率
fps=video.get(cv2.CAP_PROP_FPS)
#获取视频长宽大小
video_width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
video_height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)
#输出已经标注数据的视频
# videoWriter=cv2.VideoWriter("OUT_Test.mp4",cv2.VideoWriter_fourcc('X','2','6','4'),fps,(video_width,video_height))
print(fps)
print(video_width)
print(video_height)
i=0
while True:
    #读取视频帧
    ret,frame=video.read()
    #缩小帧大小加快计算速度
    frame=cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
    #拷贝没有标注的备份
    cop=frame.copy()
    #高斯滤波过滤高频噪声
    blur = cv2.GaussianBlur(frame, (9, 9),0)
    #处理轮廓图相当于高通滤波
    canny=cv2.Canny(blur,150,200)
    #轮廓检测
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:       
        area = cv2.contourArea(cnt)
        #判断轮廓大小过滤杂乱图形
        if area > 100:
            #绘制边缘检测点
            cv2.drawContours(frame, cnt, -1, (0, 0, 255), 4)
            #计算周长可以根据周长数据再次过滤一遍
            peri=cv2.arcLength(cnt,True)
            #将曲线转换为折线点方便判断图形形状
            vertices = cv2.approxPolyDP(cnt, peri * 0.02, True)
            #获取折线图点数可以再做一次过滤
            corners=len(vertices)
            #计算出最小边框
            x, y, w, h = cv2.boundingRect(vertices)
            #计算出目标中心点
            center_x=int(x+w/2)
            center_y=int(y+h/2)
            #计算并绘制出最小边框以及图像重心占屏幕比例与折线点数量
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame,str(corners),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
            cv2.putText(frame,str(round(center_x/video_width,2))+"|"+str(round(center_y/video_height,2)),(center_x,center_y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
            #将识别到的图块提取出来方便后面机器学习图像比对用也可以不提取使用传统轮廓检测方式来实现自动瞄准
            if area>150:  
                newimg=cop[y:y+h,x:x+w]
                if len(newimg)>10:
                    newimgname=str(i)+".jpg"
                    print(newimg)
                    cv2.imwrite(newimgname,newimg)              
                    i=i+1
    cv2.imshow("Video",frame)
    #输出带有标注的视频
    # videoWriter.write(frame)
    if not ret:
        break
    if cv2.waitKey(int(1000/fps))==ord('q'):
        break