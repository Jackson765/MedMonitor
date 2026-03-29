from comms import runBot
from camera import startCam
import threading

t1 = threading.Thread(target=runBot)
t2 = threading.Thread(target=startCam)
t1.daemon = True
t2.daemon = True

t2.start()
runBot()