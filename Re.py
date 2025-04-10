from paho.mqtt import client as mqtt_client
import base64
from datetime import datetime
import os
#import MQTT_Public
import Code
import cv2
import numpy as np
# Cấu hình HiveMQ
BROKER = "b4986a030491478188bb6f138c2fed13.s1.eu.hivemq.cloud"
PORT = 8883  # Cổng SSL/TLS
USERNAME = "chisnhaan"
PASSWORD = "0948108474Nh@n"
TOPIC = "image/upload"

# Thư mục lưu ảnh
SAVE_FOLDER = "d:/HCMUT/Year_2024-2025/242/DA1/Source/received_images"
os.makedirs(SAVE_FOLDER, exist_ok=True)

#CODE_FILE = "d:/HCMUT/Year_2024-2025/242/DA1/Source/Code.py"

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Successfully connected to HiveMQ")
        else:
            print(f"Failed to connect, return code {rc}")
    
    client = mqtt_client.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()  # Dùng TLS cho kết nối an toàn
    client.on_connect = on_connect
    client.connect(BROKER, PORT,6000)
    return client

def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        try:
            image_data = msg.payload.decode()
            img = image_data.encode('ascii')
            
            now = datetime.now()
            filename = now.strftime("%Y-%m-%d_%H%M%S.jpg")
            filepath = os.path.join(SAVE_FOLDER, filename)
            
            with open(filepath, 'wb') as f:
                final_img = base64.b64decode(img)
                f.write(final_img)
                # Chuyển dữ liệu ảnh thành numpy array
                image_np = np.frombuffer(final_img, dtype=np.uint8)
                # Giải mã ảnh thành định dạng OpenCV (BGR)
                image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
                print ("Image is processing")
                Crab = Code.image_processing(image)
                print("The number of Crab",Crab)
                client.publish("image/Crab",Crab)
                print("Waitinng for next image")
        except Exception as e:
            print(f" Error processing image: {e}")
    
    client.subscribe(TOPIC)
    client.on_message = on_message

def main():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()

if __name__ == "__main__":
    main()
