from Deployment.usefulFunctions import *
from time import time
import signal

def handler(signum, frame):
    msg = "\nCTRL C detected, exiting program..."
    print(msg, end="", flush=True)
    print(f"\nAverage time is: {sum(facialLandmarkTime) / len(facialLandmarkTime)}")
    exit(1)

signal.signal(signal.SIGINT, handler)
facialLandmarkTime = []
timer = time()

while True:
    (rects, gray, image) = getRect()
    timeneeded = time() - timer
    facialLandmarkTime.append(timeneeded)
    print(f"Time needed for facial landmark is {timeneeded}")
    timer = time()
        

