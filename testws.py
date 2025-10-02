import socketio
import time

sio_fhr = socketio.Client()

@sio_fhr.event(namespace="/ws/records/fhr")
def connect():
    print("[CLIENT] Connected to FHR")

@sio_fhr.on("connected", namespace="/ws/records/fhr")
def on_connected(data):
    print("[CLIENT] handshake:", data)

@sio_fhr.on("fhr:snapshot", namespace="/ws/records/fhr")
def on_snapshot(data):
    print("[CLIENT] snapshot items:", len(data["items"]))

@sio_fhr.on("fhr:new", namespace="/ws/records/fhr")
def on_new(data):
    print("[CLIENT] new:", data)

@sio_fhr.on("disconnect", namespace="/ws/records/fhr")
def on_disc():
    print("[CLIENT] disconnected (emulator likely stopped)")

@sio_fhr.on("emulator:finished", namespace="/ws/records/fhr")
def on_finished(data):
    print("[CLIENT] emulator finished:", data)
    sio_fhr.disconnect(namespace="/ws/records/fhr")

sio_fhr.connect("http://localhost:5050/ws/records/fhr?visit_id=1",
                transports=["polling"])
print("[CLIENT] Waiting events...")
sio_fhr.wait()

