# pyrefly: ignore [missing-import]
import cv2
import os

def generate_aruco_urdf(marker_id, filename):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_grid = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 6)
    
    xml = f'<?xml version="1.0"?>\n<robot name="{filename.split(".")[0]}">\n  <link name="base_link">\n'
    
    # 1. Beyaz Zemin
    xml += '    <visual>\n      <origin xyz="0 0 0.005" rpy="0 0 0"/>\n      <geometry>\n        <box size="1.2 1.2 0.01"/>\n      </geometry>\n      <material name="white"><color rgba="1 1 1 1"/></material>\n    </visual>\n'
    xml += '    <collision>\n      <origin xyz="0 0 0.005" rpy="0 0 0"/>\n      <geometry>\n        <box size="1.2 1.2 0.01"/>\n      </geometry>\n    </collision>\n'
    xml += '    <inertial>\n      <mass value="50.0"/>\n      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>\n    </inertial>\n'

    # 2. Siyah Pikseller (ID'ye göre şekillenir)
    pixel_size = 0.15
    start_x = - (6 * pixel_size) / 2 + (pixel_size / 2)
    start_y = (6 * pixel_size) / 2 - (pixel_size / 2)

    for i in range(6):
        for j in range(6):
            if marker_grid[i, j] == 0: 
                x = start_x + j * pixel_size
                y = start_y - i * pixel_size
                xml += f'    <visual>\n      <origin xyz="{x:.3f} {y:.3f} 0.011" rpy="0 0 0"/>\n      <geometry>\n        <box size="{pixel_size:.3f} {pixel_size:.3f} 0.002"/>\n      </geometry>\n      <material name="black"><color rgba="0.05 0.05 0.05 1"/></material>\n    </visual>\n'

    xml += '  </link>\n</robot>'

    with open(filename, "w") as f:
        f.write(xml)
    print(f"✅ {filename} (ID: {marker_id}) başarıyla oluşturuldu!")

# ID=0 (Sağa Dönüş), ID=1 (İniş), ID=2 (Sola Dönüş), ID=3 (İleri)
generate_aruco_urdf(0, "aruco_turn_right.urdf")
generate_aruco_urdf(1, "aruco_land.urdf")
generate_aruco_urdf(2, "aruco_turn_left.urdf")
generate_aruco_urdf(3, "aruco_forward.urdf")
