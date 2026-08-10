import genesis as gs
import numpy as np

gs.init(backend=gs.cpu)
scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
drone = scene.add_entity(gs.morphs.URDF(file='helionv4.urdf', pos=(0.0, 0.0, 0.05)))
scene.build()

print("drone.n_dofs:", drone.n_dofs)
print("drone.get_pos():", drone.get_pos())
print("drone.get_quat():", drone.get_quat())
print("drone.get_dofs_velocity():", drone.get_dofs_velocity())
