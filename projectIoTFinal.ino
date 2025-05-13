#include <WiFi.h>
#include <PubSubClient.h>

const char* ssid = "YOUR_Wi-Fi_SSID";
const char* password = "YOUR_Wi-Fi_Passwd";
const char* mqtt_server = "YOUR_MQTT_BROKER_IP_ADDRESS";

// GPIO Pins
#define GREEN_LED_PIN     27
#define RED_LED_PIN       26
#define PHOTOSENSOR_PIN   13
#define GREEN_BUTTON_PIN  32
#define RED_BUTTON_PIN    33

WiFiClient espClient;
PubSubClient client(espClient);

enum State { IDLE, WAIT_CAR_PASS, WAIT_DELAY };
State currentState = IDLE;

unsigned long delayStart = 0;
bool lastPhotoState = false;

void setup_wifi() {
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println(" connected!");
}

void setLights(bool green, bool red) {
  digitalWrite(GREEN_LED_PIN, green ? LOW : HIGH);
  digitalWrite(RED_LED_PIN, red ? LOW : HIGH);
  Serial.print("Set Lights → Green: ");
  Serial.print(green);
  Serial.print(", Red: ");
  Serial.println(red);
}

void callback(char* topic, byte* payload, unsigned int length) {
  String msg;
  for (int i = 0; i < length; i++) msg += (char)payload[i];
  msg.trim();

  Serial.print("MQTT received on ");
  Serial.print(topic);
  Serial.print(": ");
  Serial.println(msg);

  if (String(topic) == "license_plate/detection" && msg.length() > 0) {
    // ป้ายทะเบียนถูกตรวจพบ
    setLights(true, false);
    currentState = WAIT_CAR_PASS;
    Serial.println("License plate detected → GREEN light ON → Waiting for car...");
  }
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    if (client.connect("ESP32_Controller")) {
      Serial.println(" connected.");
      client.subscribe("license_plate/detection");
      Serial.println("Subscribed to license_plate/detection");
    } else {
      Serial.print(" failed, rc=");
      Serial.print(client.state());
      Serial.println(" retrying in 5 sec");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(GREEN_LED_PIN, OUTPUT);
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(PHOTOSENSOR_PIN, INPUT);
  pinMode(GREEN_BUTTON_PIN, INPUT_PULLUP);
  pinMode(RED_BUTTON_PIN, INPUT_PULLUP);

  setLights(false, true); // เริ่มด้วยไฟแดง

  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
}

void loop() {
  if (!client.connected()) reconnect();
  client.loop();

  bool photoBlocked = digitalRead(PHOTOSENSOR_PIN) == LOW;

  switch (currentState) {
    case WAIT_CAR_PASS:
      if (!photoBlocked && lastPhotoState) {
        // รถเพิ่งผ่าน (จาก LOW → HIGH)
        delayStart = millis();
        currentState = WAIT_DELAY;
        Serial.println("Car passed → starting 4s timer...");
      }
      break;

    case WAIT_DELAY:
      if (millis() - delayStart >= 4000) {
        setLights(false, true);
        currentState = IDLE;
        Serial.println("4s passed → RED light ON");
      }
      break;

    case IDLE:
    default:
      break;
  }

  lastPhotoState = photoBlocked;
  delay(100);
}