from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import os
from playsound import playsound
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load YOLOv3 model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
weapon_classes = ["gun", "knife", "sword"]

# Alarm sound file
alarm_sound = "alarm.mp3"
alarm_active = False

# Owner's email
OWNER_EMAIL = "n.sharam9561@gmail.com"  # Replace with actual email
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USERNAME = "n.sharma956@gmail.com"
EMAIL_PASSWORD = "ettc vpml bbax bwd"

# Shared state for real-time sentiment text
current_text = "Everything seems fine"


# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')  # Just display the video feed and main content


# Video stream generator
def generate_video_stream():
    global alarm_active, current_text
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())

        weapon_detected = False

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] in weapon_classes:
                    weapon_detected = True
                    center_x, center_y, w, h = detection[0:4] * np.array([width, height, width, height])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    cv2.rectangle(frame, (x, y), (x + int(w), y + int(h)), (0, 0, 255), 2)
                    label = f"{classes[class_id]}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if weapon_detected and not alarm_active:
            alarm_active = True
            threading.Thread(target=playsound, args=(alarm_sound,)).start()
            threading.Thread(target=send_notification, args=(current_text, frame.copy())).start()

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Stop the alarm
@app.route('/stop_alarm', methods=['POST'])
def stop_alarm():
    global alarm_active
    alarm_active = False
    return "Alarm stopped successfully"


# Update sentiment text
@app.route('/update_sentiment', methods=['POST'])
def update_sentiment():
    global current_text
    current_text = request.form.get('text', "Everything seems fine")
    return f"Sentiment text updated to: {current_text}"


# Send email notification with image
def send_notification(threat_text, frame):
    try:
        image_path = "threat.jpg"
        cv2.imwrite(image_path, frame)

        msg = MIMEMultipart()
        msg['Subject'] = "Security Alert: Weapon Detected"
        msg['From'] = EMAIL_USERNAME
        msg['To'] = OWNER_EMAIL

        body = MIMEText("Intruder with weapon detected, call security.")
        msg.attach(body)

        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
        image = MIMEImage(img_data, name=os.path.basename(image_path))
        msg.attach(image)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USERNAME, OWNER_EMAIL, msg.as_string())
        os.remove(image_path)
    except Exception as e:
        print(f"Failed to send email with image: {e}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
