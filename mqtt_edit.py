import cv2
import paho.mqtt.client as mqtt
from ultralytics import YOLO
from function.helper import get_thai_character, data_province, split_license_plate_and_province
from datetime import datetime 
import time


import mysql.connector

# Load models
vehicle_model = YOLO("model/license_plate.pt")
plate_model = YOLO("model/data_plate.pt")
print(plate_model.names)

# MQTT Configuration
MQTT_BROKER = "YOUR_IP_ADDRESS"  # Replace with your MQTT broker IP address
MQTT_PORT = 1883
MQTT_TOPIC = "license_plate/detection"

# MySQL Configuration
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="barriergate"
)
cursor = db.cursor()

# Initialize MQTT Client
client = mqtt.Client() 
client.on_connect = lambda client, userdata, flags, rc, properties=None: print("Connected with reason code", rc)
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

sent_plates = set()

def get_thai_license_plate_from_rtsp(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("❌ Failed to open RTSP stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from RTSP")
            time.sleep(1)
            continue

        frame = cv2.resize(frame, (1280, 720))
        vehicle_results = vehicle_model(frame, conf=0.3, verbose=False)
        detected_classes = []

        for result in vehicle_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                car_roi = frame[y1:y2, x1:x2]
                plate_results = plate_model(car_roi, conf=0.3, verbose=False)
                plates = []
                for plate in plate_results:
                    for plate_box in plate.boxes:
                        px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                        px1 += x1
                        px2 += x1
                        py1 += y1
                        py2 += y1
                        plates.append((px1, plate_box.cls, (px1, py1, px2, py2)))

                plates.sort(key=lambda x: x[0])
                for plate in plates:
                    px1, cls, (x1_plate, y1_plate, x2_plate, y2_plate) = plate
                    cv2.rectangle(frame, (x1_plate, y1_plate), (x2_plate, y2_plate), (255, 255, 0), 2)
                    clsname = plate_model.names[int(cls)]
                    detected_classes.append(clsname)

        for item in detected_classes:
            if item in data_province:
                detected_classes.remove(item)
                detected_classes.append(item)

        combined_text = "".join(get_thai_character(newval) for newval in detected_classes)
        license_plate, province = split_license_plate_and_province(combined_text)
        print("ทะเบียนรถ:", license_plate, "จังหวัด:", province)

       
       

        if license_plate and province and license_plate not in sent_plates:
             # ตรวจสอบทะเบียนในฐานข้อมูล
            query = "SELECT car_registration, province FROM car WHERE car_registration = %s AND province = %s"
            cursor.execute(query, (license_plate, province))
            result = cursor.fetchone()


            if result:
                print("✅ พบทะเบียนในระบบ:", result)
                date = datetime.now().strftime("%Y-%m-%d ")
                time_now = datetime.now().strftime("%H:%M:%S")
                data = f"ทะเบียนรถ : {license_plate} \n จังหวัด : {province} \n {date} \n {time_now} "
                client.publish(MQTT_TOPIC, data)
                print("✅ ส่งข้อมูลไป MQTT:", data)
                print("✅ ส่งข้อมูล 'เปิด' ไป MQTT:")
                sent_plates.add(license_plate)
            else:
                print("❌ ไม่พบทะเบียนในระบบ")
                data2=f'ปิด'
                client.publish(MQTT_TOPIC, data)
                print("❌ ส่งข้อมูล 'ปิด' ไป MQTT:",data)
    
    

        cv2.imshow("License Plate Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
             
# Run
def main():
    rtsp_url = "rtsp://YOUR_RTSP_STREAM_URL"
    # Replace with your actual RTSP stream URL
    get_thai_license_plate_from_rtsp(rtsp_url)

if __name__ == "__main__":
    main()