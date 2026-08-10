import genesis as gs
import numpy as np
import cv2

gs.init(backend=gs.cpu)
scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
# Create red sphere at Y=1.0
sphere = scene.add_entity(gs.morphs.Sphere(pos=(0.0, 1.0, 0.5), radius=0.5))

cam = scene.add_camera(res=(640, 480), pos=(0, 0, 5), lookat=(0, 0, 0), up=(1, 0, 0))
scene.build()

# Force red color
# Actually, the easiest way is to print the position of sphere in camera coordinates, but we can just use cv2
scene.step()
img = cam.render()[0]
if img.dtype == np.float32: img = (img * 255).astype(np.uint8)
cv2.imwrite("test_cam.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print("Saved test_cam.jpg")
