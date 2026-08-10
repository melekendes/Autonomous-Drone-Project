import genesis as gs
import numpy as np

gs.init(backend=gs.cpu)
scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
cam = scene.add_camera(res=(640, 480), pos=(0, 0, 2), lookat=(0, 0, 0))
scene.build()
cam.set_pose(pos=np.array([0,0,2]), lookat=np.array([0,0,0]), up=np.array([1,0,0]))
