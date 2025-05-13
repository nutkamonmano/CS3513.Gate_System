import cv2
import paho.mqtt.client as mqtt
from ultralytics import YOLO
from function.helper import get_thai_character, data_province, split_license_plate_and_province

# Load models
vehicle_model = YOLO("model/license_plate.pt")  # Vehicle detection model
plate_model = YOLO("model/data_plate.pt")  # License plate detection model
print(plate_model.names)

# MQTT Configuration
MQTT_BROKER = "YOUR_IP_ADDRESS"  # Replace with your MQTT broker IP address
MQTT_PORT = 1883
MQTT_TOPIC = "license_plate/detection" # Topic to publish license plate data ใช้ topic ตามนี้ใน node-red

# Initialize MQTT Client
client = mqtt.Client()
client.on_connect = lambda client, userdata, flags, rc: print("Connected with result code", rc)
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()  # Start MQTT loop

# เก็บทะเบียนที่ส่งไปแล้ว
sent_plates = set()

def get_thai_license_plate_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))

        # Detect vehicles
        vehicle_results = vehicle_model(frame, conf=0.3, verbose=False)
        if not vehicle_results:
            continue  # Skip if no vehicles detected

        detected_classes = []
        
        for result in vehicle_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green = vehicle
                
                # Crop vehicle region
                car_roi = frame[y1:y2, x1:x2]
                plate_results = plate_model(car_roi, conf=0.3, verbose=False)
                if not plate_results:
                    continue  # Skip if no plates detected

                plates = []
                for plate in plate_results:
                    for plate_box in plate.boxes:
                        px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                        px1, px2 = px1 + x1, px2 + x1
                        py1, py2 = py1 + y1, py2 + y1
                        plates.append((px1, plate_box.cls, (px1, py1, px2, py2)))

                # Sort plates by X coordinate
                plates.sort(key=lambda x: x[0])
                
                for plate in plates:
                    px1, cls, (x1_plate, y1_plate, x2_plate, y2_plate) = plate
                    cv2.rectangle(frame, (x1_plate, y1_plate), (x2_plate, y2_plate), (255, 255, 0), 2)  # Blue = license plate
                    clsname = plate_model.names[int(cls)]
                    detected_classes.append(clsname)

        # Process detected characters
        for item in detected_classes:
            if item in data_province:
                detected_classes.remove(item)
                detected_classes.append(item)

        combined_text = "".join(get_thai_character(newval) for newval in detected_classes)
        license_plate, province = split_license_plate_and_province(combined_text)
        print("ทะเบียนรถ:", license_plate, "จังหวัด:", province)

        # ส่ง MQTT เฉพาะทะเบียนที่ยังไม่เคยส่ง
        if license_plate and province and license_plate not in sent_plates:
            data = f'{{"license_plate": "{license_plate}", "province": "{province}"}}'
            client.publish(MQTT_TOPIC, data)
            print("✅ ส่งข้อมูลไป MQTT:", data)
            sent_plates.add(license_plate)  # บันทึกว่าทะเบียนนี้ส่งแล้ว

        # Show frame
        cv2.imshow("License Plate Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

# Run function
get_thai_license_plate_from_video("video/video1.avi")