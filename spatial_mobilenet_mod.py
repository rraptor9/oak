#!/usr/bin/env python3
from pathlib import Path
import sys, time, cv2, depthai as dai, numpy as np
import socket

##configuracion del receptor
socka = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
dest_ip = "192.168.1.44" # IP del receptor
dest_port = 5005
mensaje = f"iniciand"
socka.sendto(mensaje.encode(), (dest_ip, dest_port))

nnBlobPath = str((Path(__file__).parent / './models/custom_model.blob').resolve())
if len(sys.argv) > 1:
    nnBlobPath = sys.argv[1]
if not Path(nnBlobPath).exists():
    raise FileNotFoundError(f'No se encontró el blob: {nnBlobPath}')

labelMap = ["unknown"]
syncNN = True

pipeline = dai.Pipeline()

# ----- Fuentes -----
camRgb   = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight= pipeline.create(dai.node.MonoCamera)
stereo   = pipeline.create(dai.node.StereoDepth)
spatial  = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)

# ----- Salidas -----
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN  = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")

# ----- Config cámaras -----
camRgb.setPreviewSize(384,384)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

for mono, sock in [(monoLeft,"left"), (monoRight,"right")]:
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono.setCamera(sock)

# ----- Config estéreo -----
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setSubpixel(True)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

# ----- Config red -----
spatial.setBlobPath(nnBlobPath)
spatial.setConfidenceThreshold(0.5)
spatial.setBoundingBoxScaleFactor(0.5)
spatial.setDepthLowerThreshold(100)
spatial.setDepthUpperThreshold(5000)
spatial.input.setBlocking(False)

# ----- Enlaces -----
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
camRgb.preview.link(spatial.input)
stereo.depth.link(spatial.inputDepth)

if syncNN:
    spatial.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

spatial.out.link(xoutNN.input)

# ---------- EJECUCIÓN ----------
with dai.Device(pipeline) as device:
    previewQ = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    detQ     = device.getOutputQueue("detections", maxSize=4, blocking=False)

    t0, counter, fps = time.monotonic(), 0, 0
    while True:
        frame   = previewQ.get().getCvFrame()
        dets    = detQ.get().detections

        h, w = frame.shape[:2]
        for d in dets:
            x1,y1 = int(d.xmin*w), int(d.ymin*h)
            x2,y2 = int(d.xmax*w), int(d.ymax*h)
            
            lbl   = labelMap[d.label] if d.label < len(labelMap) else d.label
            x = int(d.spatialCoordinates.x)
            y = int(d.spatialCoordinates.y)
            z = int(d.spatialCoordinates.z)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.putText(frame, str(lbl), (x1+8, y1+20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
            cv2.putText(frame, f"{d.confidence*100:.2f}", (x1+8, y1+35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
            cv2.putText(frame, f"X:{x}", (x1+8, y1+50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
            cv2.putText(frame, f"Y:{y}", (x1+8, y1+65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
            cv2.putText(frame, f"Z:{z}", (x1+8, y1+80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))

            print(f"localizado X = {x} Y = {y} Z = {z}")
            mensaje = f"label:{lbl}, x:{x}, y:{y}, z:{z}"
            socka.sendto(mensaje.encode(), (dest_ip, dest_port))

        # FPS
        counter += 1
        if time.monotonic() - t0 >= 1:
            fps, counter, t0 = counter/(time.monotonic()-t0), 0, time.monotonic()
        cv2.putText(frame,f"NN fps: {fps:.2f}",(2,h-4),cv2.FONT_HERSHEY_TRIPLEX,0.4,(255,255,255))

        cv2.imshow("qr detector",frame)
        if cv2.waitKey(1) == ord('q'):
            break
