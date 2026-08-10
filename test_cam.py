import genesis as gs
import numpy as np

gs.init(backend=gs.cpu)
scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
cam = scene.add_camera(res=(640, 480), pos=(0, 0, 2), lookat=(0, 0, 0))
scene.build()
scene.step()
res = cam.render()
print(type(res), len(res) if isinstance(res, tuple) else "Not a tuple")
if isinstance(res, tuple):
    for i, r in enumerate(res):
        print(f"res[{i}] shape: {getattr(r, 'shape', 'No shape')}")
else:
    print("res shape:", getattr(res, 'shape', 'No shape'))
