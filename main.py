import csv
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from tkinter import *
from tkinter import ttk, Tk, Label

path = 'ImagesAttendance'  # path of images
images = []
classNames = []
myList = os.listdir(path)
# print(myList)
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])


# print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            tString = now.strftime('%H:%M:%S')
            dString = now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tString},{dString}')


encodeListKnow = findEncodings(images)


# print('Encoding Complete')


def cam():
    labelcam = Label(window)
    labelcam.pack(side=TOP)
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 255), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
                markAttendance(name)
        cv2.imshow('web Cam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()


window = Tk()
window.title("AI Attendance")
window.geometry('650x650')
window.configure(background="white")
label = Label(window, text="AI ATTENDANCE", width=20, height=2, relief='flat', anchor=CENTER, font=('Ivy 15 bold'),
              fg="brown", activebackground="brown")
label.pack(side=TOP, pady=5)


def clicked():
    cam()


def export():
    with open('Attendance.csv', 'r') as firstfile, open('ExportAttendance.csv', 'w') as secondfile:
        for line in firstfile:
            secondfile.write(line)
    f = open("Attendance.csv", "w")
    f.truncate()
    f.close()


button = ttk.Button(window, text='Take Attendance', command=clicked)
button.pack(side=LEFT, expand=TRUE)

button2 = ttk.Button(window, text='Export CSV', command=export)
button2.pack(side=RIGHT, expand=TRUE)

window.mainloop()
