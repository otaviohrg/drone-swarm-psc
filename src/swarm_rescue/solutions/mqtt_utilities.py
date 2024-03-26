import random
import time

from paho.mqtt import client as mqtt_client

BROKER = '833debc11125483589a6c5eec37e2c0e.s1.eu.hivemq.cloud'
PORT = 8883
TOPIC = "swarm-rescue"
USERNAME = 'HIVEMQ_user'

FIRST_RECONNECT_DELAY = 1
RECONNECT_RATE = 2
MAX_RECONNECT_COUNT = 12
MAX_RECONNECT_DELAY = 60

class MyDroneMQTT():

    def __init__(self):
        self.client_id = f'drone-{random.randint(0, 1000)}'
        self.client = self.connect_mqtt()
        self.client.loop_start()
        time.sleep(1)
        if self.client.is_connected():
            print(f"{self.client_id} connected :)")
        else:
            self.client.loop_stop()

    def connect_mqtt(self):
        self.client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION1, self.client_id)
        self.client.username_pw_set(USERNAME)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(BROKER, PORT, keepalive=120)
        self.client.on_disconnect = self.on_disconnect
        return self.client

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0 and client.is_connected():
            client.subscribe(TOPIC)
        else:
            print(f'Failed to connect, return code {rc}')

    def on_disconnect(self, client, userdata, rc):
        #print("Disconnected with result code: %s", rc)
        reconnect_count, reconnect_delay = 0, FIRST_RECONNECT_DELAY
        while reconnect_count < MAX_RECONNECT_COUNT:
            #print("Reconnecting in %d seconds...", reconnect_delay)
            time.sleep(reconnect_delay)

            try:
                client.reconnect()
                #print("Reconnected successfully!")
                return
            except Exception as err:
                print("%s. Reconnect failed. Retrying...", err)

            reconnect_delay *= RECONNECT_RATE
            reconnect_delay = min(reconnect_delay, MAX_RECONNECT_DELAY)
            reconnect_count += 1
        #print("Reconnect failed after %s attempts. Exiting...", reconnect_count)

    def on_message(self, client, userdata, msg):
        pass
        #print(f'Received `{msg.payload.decode()}` from `{msg.topic}` topic')

    def publish(self, message):
        num_tries = 5
        while num_tries > 0:
            result = self.client.publish(TOPIC, message)
            # result: [0, 1]
            status = result[0]
            if status == 0:
                #print(f"Sent `{message}` to topic `{TOPIC}`")
                num_tries = 0
            else:
                #print(f"Failed to send message to topic {TOPIC}")
                num_tries -= 1