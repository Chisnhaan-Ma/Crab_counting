import time
import base64
from paho.mqtt import client as mqtt_client
#import Code 
# Cấu hình HiveMQ
BROKER = "b4986a030491478188bb6f138c2fed13.s1.eu.hivemq.cloud"
PORT = 8883  # Cổng SSL/TLS
USERNAME = "chisnhaan"
PASSWORD = "0948108474Nh@n"
TOPIC = "image/upload"
CLIENT_ID = "clientID123"  # Bạn có thể đặt tên client tùy ý

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("✅ Kết nối thành công đến HiveMQ!")
        else:
            print(f"❌ Kết nối thất bại, mã lỗi: {rc}")

    client = mqtt_client.Client(CLIENT_ID)
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()  # Bật TLS cho kết nối bảo mật
    client.on_connect = on_connect
    client.connect(BROKER, PORT, 60)
    return client

def publish(client):
    try:
        with open("./1.jpg", "rb") as file:
            file_content = file.read()
            base64_content = base64.b64encode(file_content)

            result = client.publish(TOPIC, base64_content)
            msg_status = result.rc  # Kiểm tra trạng thái gửi

            if msg_status == 0:
                print(f"✅ Ảnh đã gửi thành công đến chủ đề '{TOPIC}'")
            else:
                print(f"❌ Gửi ảnh thất bại, mã lỗi: {msg_status}")

    except Exception as e:
        print(f"❌ Lỗi khi đọc/gửi ảnh: {e}")

def main():
    client = connect_mqtt()
    client.loop_start()  # Bắt đầu vòng lặp MQTT
    time.sleep(2)  # Đợi kết nối ổn định

    publish(client)

    time.sleep(5)  # Giữ kết nối một lúc
    client.loop_stop()  # Dừng vòng lặp MQTT
    client.disconnect()  # Ngắt kết nối

if __name__ == "__main__":
    main()
