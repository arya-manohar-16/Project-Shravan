// ─────────────────────────────────────────────────────────────────────────────
// NodeMCU ESP8266: gesture chair over MQTT (receives commands from the phone web app)
//
// Library needed: "PubSubClient" by Nick O'Leary (Arduino IDE > Library Manager)
// Board: NodeMCU 1.0 (ESP-12E Module)
//
// The ESP joins a Wi-Fi network that has internet (your phone's hotspot is fine,
// set it to 2.4 GHz). The phone web app publishes F / B / L / R / S to a public
// MQTT broker and this sketch subscribes to the same topic.
// ─────────────────────────────────────────────────────────────────────────────

#include <ESP8266WiFi.h>
#include <PubSubClient.h>

// ── Edit these four lines ─────────────────────────────────────────────────────
const char* WIFI_SSID = "YOUR_HOTSPOT_NAME";
const char* WIFI_PASS = "YOUR_HOTSPOT_PASSWORD";
const char* LINK_CODE = "PASTE_THE_CODE_FROM_THE_WEB_APP";   // same code as in the app's Settings
const char* MQTT_HOST = "broker.hivemq.com";                 // EMQX option: "broker.emqx.io"
const uint16_t MQTT_PORT = 1883;

// If no command arrives for this long, the motors stop.
const unsigned long CMD_TIMEOUT_MS = 800;

// ── Motor pins (same as your current sketch) ──────────────────────────────────
const int IN1 = 5;    // D1: left motor
const int IN2 = 4;    // D2: left motor
const int IN3 = 14;   // D5: right motor
const int IN4 = 12;   // D6: right motor

const int SPEED      = 300;   // ESP8266 analogWrite range is 0-1023
const int TURN_SPEED = 200;

WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);
String cmdTopic, statusTopic;

char lastApplied = 'S';
unsigned long lastCmdTime = 0, lastTry = 0, lastStatus = 0;

// ── Motor functions ───────────────────────────────────────────────────────────
void moveForward()  { analogWrite(IN1, SPEED);      analogWrite(IN2, 0);
                      analogWrite(IN3, 0);          analogWrite(IN4, SPEED);      Serial.println("Action: FORWARD"); }
void moveBackward() { analogWrite(IN1, 0);          analogWrite(IN2, SPEED);
                      analogWrite(IN3, SPEED);      analogWrite(IN4, 0);          Serial.println("Action: BACKWARD"); }
void turnRight()    { analogWrite(IN1, TURN_SPEED); analogWrite(IN2, 0);
                      analogWrite(IN3, 0);          analogWrite(IN4, 0);          Serial.println("Action: RIGHT"); }
void turnLeft()     { analogWrite(IN1, 0);          analogWrite(IN2, 0);
                      analogWrite(IN3, 0);          analogWrite(IN4, TURN_SPEED); Serial.println("Action: LEFT"); }
void stopMotors()   { analogWrite(IN1, 0); analogWrite(IN2, 0);
                      analogWrite(IN3, 0); analogWrite(IN4, 0);                   Serial.println("Action: STOP"); }

// ── Incoming command ──────────────────────────────────────────────────────────
void onMessage(char* topic, byte* payload, unsigned int length) {
  if (length == 0) return;
  lastCmdTime = millis();                 // every message, even a repeat, counts as a heartbeat
  char c = (char)payload[0];
  if (c == lastApplied) return;           // same command again: motors are already doing it
  lastApplied = c;

  switch (c) {
    case 'F': moveForward();  break;
    case 'B': moveBackward(); break;
    case 'L': turnRight();    break;      // swapped on purpose, same as your current sketch
    case 'R': turnLeft();     break;
    case 'S': stopMotors();   break;
    default:  stopMotors(); lastApplied = 'S'; break;
  }
}

void forceStop() {
  if (lastApplied != 'S') { stopMotors(); lastApplied = 'S'; }
}

// ── Connections ───────────────────────────────────────────────────────────────
void ensureMqtt() {
  if (mqtt.connected()) return;
  forceStop();                            // never drive while the link is down
  if (millis() - lastTry < 2000) return;
  lastTry = millis();

  String id = "chair-esp-" + String(ESP.getChipId(), HEX);
  if (mqtt.connect(id.c_str())) {
    mqtt.subscribe(cmdTopic.c_str());
    Serial.println("MQTT connected");
  } else {
    Serial.print("MQTT failed, rc="); Serial.println(mqtt.state());
  }
}

void setup() {
  Serial.begin(115200);
  // Generate or read the secure link code (Update this to output your secure 20+ char string)
  String linkCode = "YOUR_SECURE_20_CHAR_CODE_HERE"; 
  
  Serial.println();
  Serial.println("================================");
  Serial.print("Website Link Code: ");
  Serial.println(linkCode);
  Serial.println("Enter this code on the website.");
  Serial.println("================================");
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  stopMotors();

  cmdTopic    = String("gesturechair/") + LINK_CODE + "/cmd";
  statusTopic = String("gesturechair/") + LINK_CODE + "/status";

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print("."); }
  Serial.println();
  Serial.print("IP: "); Serial.println(WiFi.localIP());
  Serial.print("Command topic: "); Serial.println(cmdTopic);

  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(onMessage);
  mqtt.setSocketTimeout(3);               // don't block for 15 s when the broker is unreachable
  mqtt.setKeepAlive(15);
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) { forceStop(); delay(200); return; }

  ensureMqtt();
  mqtt.loop();

  // failsafe: phone stopped talking (app closed, no signal, phone asleep)
  if (lastApplied != 'S' && millis() - lastCmdTime > CMD_TIMEOUT_MS) {
    forceStop();
    Serial.println("Failsafe: no command received, STOP");
  }

  // tell the web app we're alive
  if (mqtt.connected() && millis() - lastStatus > 2000) {
    lastStatus = millis();
    mqtt.publish(statusTopic.c_str(), "online");
  }
}
