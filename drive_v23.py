import genesis as gs
import numpy as np
import cv2
import time
import random

DRONE_MASS = 18.11 / 9.81 
GRAVITY = 9.81
HOVER_PWM = 1362.0
CENTER_PWM = 1500.0

BATTERY_VOLTAGE_MAX = 22.2  
MOTOR_KV = 900.0        
MAX_RPM = BATTERY_VOLTAGE_MAX * MOTOR_KV  

C_T = 3.4618e-7

TARGET_ALTITUDE = 1.5

MIN_RC_VALUE = 1400            
MAX_RC_VALUE = 1600            

class SimulatedSensors:
    def __init__(self):
        self.gps_noise_std = 0.02  
        self.baro_noise_std = 0.01 
        self.z_drift = 0.0         

    def read_sensors(self, true_pos, dt):
        self.z_drift += np.random.normal(0, 0.0005) * dt 
        noisy_x = true_pos[0] + np.random.normal(0, self.gps_noise_std)
        noisy_y = true_pos[1] + np.random.normal(0, self.gps_noise_std)
        noisy_z = true_pos[2] + np.random.normal(0, self.baro_noise_std) + self.z_drift
        return np.array([noisy_x, noisy_y, noisy_z])

class VectorLowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.filtered_vector = None

    def compute(self, current_vector):
        if self.filtered_vector is None:
            self.filtered_vector = current_vector 
        else:
            self.filtered_vector = (self.alpha * current_vector) + ((1.0 - self.alpha) * self.filtered_vector)
        return self.filtered_vector

gs.init(backend=gs.cpu)
scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(0.9, 0.9, 0.9))) 

scene.add_entity(gs.morphs.URDF(file='aruco_turn_left.urdf', pos=(3.0, 0.0, 0.0)))
scene.add_entity(gs.morphs.URDF(file='aruco_turn_right.urdf', pos=(3.0, 3.0, 0.0)))
scene.add_entity(gs.morphs.URDF(file='aruco_land.urdf', pos=(6.0, 3.0, 0.0)))

# Binalar
binalar = [
    (1.5, 1.5), (1.5, -1.5), (4.5, 1.5),
    (1.5, 3.0), (1.5, 4.5), (4.5, 4.5),
    (4.5, -1.5), (6.0, 1.5), (6.0, 4.5), (7.5, 3.0)
]
for bx, by in binalar:
    scene.add_entity(gs.morphs.Box(size=(1.0, 1.0, 3.0), pos=(bx, by, 1.5)))

drone = scene.add_entity(gs.morphs.URDF(file='helionv4.urdf', pos=(0.0, 0.0, 0.05)))

camera = scene.add_camera(res=(640, 480), pos=(3.5, -3.5, 2.0), lookat=(1.0, 0.0, 0.8), fov=65)
drone_cam = scene.add_camera(res=(640, 480), pos=(0.0, 0.0, 1.0), lookat=(0.0, 0.0, 0.0), fov=65)

scene.build()

class SimulatedMSP:
    def apply_rc_to_physics(self, roll, pitch, throttle, true_z):
        throttle_pct = max(0.0, (throttle - 1000.0) / 1000.0)
        
        voltage_drop = (throttle_pct ** 2) * 2.5 
        current_voltage = BATTERY_VOLTAGE_MAX - voltage_drop
        dynamic_max_rpm = current_voltage * MOTOR_KV
        
        rpm_z = throttle_pct * dynamic_max_rpm
        force_z = C_T * (rpm_z ** 2) 

        if true_z < 0.4 and true_z > 0.0:
            ground_cushion_multiplier = 1.0 + (0.4 - true_z) * 0.6 
            force_z *= ground_cushion_multiplier
            force_z += np.random.normal(0, 0.3) 

        pitch_pct = (1500.0 - pitch) / 500.0  
        roll_pct = (1500.0 - roll) / 500.0    
        
        force_x = force_z * pitch_pct * 2.0
        force_y = force_z * roll_pct * 2.0

        kuvvet = np.zeros(drone.n_dofs)
        if drone.n_dofs >= 3:
            kuvvet[0], kuvvet[1], kuvvet[2] = force_x, force_y, force_z
        drone.control_dofs_force(kuvvet)
        
        return current_voltage

class PIDController:
    def __init__(self, kp, ki, kd, out_min, out_max):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint, self.integral, self.last_error = 0.0, 0.0, 0.0
        self.out_min, self.out_max = out_min, out_max
        self.orig_kp, self.orig_kd = kp, kd 

    def compute(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral = np.clip(self.integral + error * dt, -20.0, 20.0)
        p_out = self.kp * error
        i_out = self.ki * self.integral
        d_out = self.kd * (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error
        return np.clip(p_out + i_out + d_out, self.out_min, self.out_max)
        
    def reset_gains(self):
        self.kp, self.kd = self.orig_kp, self.orig_kd

class AutonomousFlightController:
    def __init__(self):
        self.pid_alt = PIDController(15.0, 2.5, 40.0, -100.0, 100.0)
        self.pid_pitch = PIDController(12.0, 1.2, 28.0, -25.0, 25.0) 
        self.pid_roll = PIDController(12.0, 1.2, 28.0, -25.0, 25.0)  

    def compute_rc_channels(self, current_pos, target_pos, dt, mod):
        self.pid_alt.setpoint = target_pos[2]
        base_throttle = HOVER_PWM if target_pos[2] > 0.3 else 1345.0 
        
        if mod == "INIŞ_ALGORITMASI" and current_pos[2] < 0.5:
            base_throttle = 1315.0 
            self.pid_alt.kp = 6.0  
            self.pid_alt.kd = 10.0 
        else:
            self.pid_alt.reset_gains()

        throttle_offset = self.pid_alt.compute(current_pos[2], dt)
        final_throttle = np.clip(base_throttle + throttle_offset, 1000.0, 2000.0)

        self.pid_pitch.setpoint = target_pos[0]
        pitch_offset = self.pid_pitch.compute(current_pos[0], dt)
        final_pitch = np.clip(CENTER_PWM - pitch_offset, 1000.0, 2000.0)

        self.pid_roll.setpoint = target_pos[1]
        roll_offset = self.pid_roll.compute(current_pos[1], dt)
        final_roll = np.clip(CENTER_PWM - roll_offset, 1000.0, 2000.0)
        
        return final_roll, final_pitch, final_throttle

def turn_left(direction):
    mapping = {"X+": "Y+", "Y+": "X-", "X-": "Y-", "Y-": "X+"}
    return mapping[direction]

def turn_right(direction):
    mapping = {"X+": "Y-", "Y-": "X-", "X-": "Y+", "Y+": "X+"}
    return mapping[direction]

sensors = SimulatedSensors()
pos_filter = VectorLowPassFilter(alpha=0.15) 
msp_bridge = SimulatedMSP()
fc = AutonomousFlightController()

mod = "KALKIS"
target = [0.0, 0.0, TARGET_ALTITUDE]

# Navigasyon
current_direction = "X+"
last_processed_id = -1
poshold_start = 0.0
poshold_marker_id = -1
align_start_time = 0.0
marker_lost_count = 0

# OpenCV ArUco ayarlari
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

with open("helion_v23_city_log.txt", "w") as log_file:
    print("\n🚀 Helion v23 — Şehir İçi ArUco Navigasyon (v20 Fizik Modeli)")
    
    try:
        for i in range(10000): 
            dt = 0.015
            elapsed = i * dt
            
            true_pos = drone.get_pos().numpy()
            noisy_pos = sensors.read_sensors(true_pos, dt)
            filtered_pos = pos_filter.compute(noisy_pos)
            
            # Sinematik takip kamerası
            camera.set_pose(pos=np.array([true_pos[0]+3.0, true_pos[1]-3.0, 2.0]), lookat=true_pos)
            
            # Drone alt kamerası
            drone_cam.set_pose(pos=true_pos, lookat=true_pos + np.array([0, 0, -1]), up=np.array([1, 0, 0]))
            
            scene.step()

            marker_found = False
            detected_id = -1
            cx_px = 320
            cy_px = 240
            detection_attempted = False
            
            if i % 3 == 0:
                detection_attempted = True
                res = drone_cam.render()
                img_rgb = res[0]
                if img_rgb.dtype == np.float32:
                    img_rgb = (img_rgb * 255).astype(np.uint8)
                
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                corners, ids, rejected = detector.detectMarkers(img_bgr)
                
                if ids is not None:
                    for idx, m_id in enumerate(ids):
                        m_id = m_id[0]
                        if m_id in [0, 1, 2, 3]:
                            marker_found = True
                            detected_id = m_id
                            c = corners[idx][0]
                            cx_px = int(np.mean(c[:, 0]))
                            cy_px = int(np.mean(c[:, 1]))
                            break

                    cv2.aruco.drawDetectedMarkers(img_bgr, corners, ids)
                    cv2.imwrite("aruco_city_debug.jpg", img_bgr)

                cv2.imshow("Drone Kamera", img_bgr)
                cv2.waitKey(1)
-
            if mod == "KALKIS":
                if filtered_pos[2] > TARGET_ALTITUDE - 0.2:
                    mod = "ARAMA"
                    print(f"\n🎯 {TARGET_ALTITUDE}m'ye çıkıldı. {current_direction} yönünde arama başlıyor...")
                    
            elif mod == "ARAMA":
                lead_dist = 0.15
                if current_direction == "X+":
                    target[0] = filtered_pos[0] + lead_dist
                elif current_direction == "X-":
                    target[0] = filtered_pos[0] - lead_dist
                elif current_direction == "Y+":
                    target[1] = filtered_pos[1] + lead_dist
                elif current_direction == "Y-":
                    target[1] = filtered_pos[1] - lead_dist

                target[2] = TARGET_ALTITUDE
                
                if marker_found and detected_id != last_processed_id:
                    print(f"\n👀 [{elapsed:.1f}s] ArUco ID {detected_id} görüldü! Hizalanıyor...")
                    target[0] = filtered_pos[0]
                    target[1] = filtered_pos[1]
                    mod = "HIZALANMA"
                    align_start_time = elapsed
                    marker_lost_count = 0
                    
            elif mod == "HIZALANMA":
                target[2] = TARGET_ALTITUDE

                if detection_attempted:
                    if marker_found:
                        marker_lost_count = 0
                        error_px = cx_px - 320
                        error_py = 240 - cy_px 
                        target[0] = filtered_pos[0] + error_py * 0.01
                        target[1] = filtered_pos[1] - error_px * 0.01
                        
                        if abs(error_px) < 50 and abs(error_py) < 50:
                            print(f"✅ Marker ID {detected_id} ortalandı! POSHOLD'a geçiliyor...")
                            target[0] = filtered_pos[0]
                            target[1] = filtered_pos[1]
                            mod = "POSHOLD"
                            poshold_start = elapsed
                            poshold_marker_id = detected_id
                    else:
                        marker_lost_count += 1
                        if marker_lost_count > 50:
                            print(f"⚠️ Marker kayboldu ({marker_lost_count} miss), ARAMA'ya dönülüyor...")
                            mod = "ARAMA"

                if elapsed - align_start_time > 10.0:
                    print(f"⚠️ Hizalanma zaman aşımı, ARAMA'ya dönülüyor...")
                    mod = "ARAMA"

            elif mod == "POSHOLD":
                target[2] = TARGET_ALTITUDE

                if elapsed - poshold_start > 4.0:
                    mid = poshold_marker_id

                    if mid == 2:  
                        new_dir = turn_left(current_direction)
                        print(f"\n↩️ Sola dönülüyor: {current_direction} → {new_dir}")
                        current_direction = new_dir
                        mod = "ARAMA"

                    elif mid == 0:  
                        new_dir = turn_right(current_direction)
                        print(f"\n↪️ Sağa dönülüyor: {current_direction} → {new_dir}")
                        current_direction = new_dir
                        mod = "ARAMA"

                    elif mid == 3:  
                        print(f"\n⬆️ İleri devam ediliyor.")
                        mod = "ARAMA"

                    elif mid == 1:  
                        print(f"\n🔽 İniş başlatılıyor!")
                        mod = "INIŞ_ALGORITMASI"

                    last_processed_id = mid

            elif mod == "INIŞ_ALGORITMASI":
                target[2] = 0.05
                
                if marker_found and detected_id == 1:
                    error_px = cx_px - 320
                    error_py = 240 - cy_px
                    target[0] = filtered_pos[0] + error_py * 0.001
                    target[1] = filtered_pos[1] - error_px * 0.001
            
            roll_pwm, pitch_pwm, throttle_pwm = fc.compute_rc_channels(filtered_pos, target, dt, mod)
            current_voltage = msp_bridge.apply_rc_to_physics(roll_pwm, pitch_pwm, throttle_pwm, true_pos[2])
            
            if i % 40 == 0:
                print(f"⏱️ {elapsed:05.1f}s | Mod: {mod:15} | Yön: {current_direction} | "
                      f"X: {filtered_pos[0]:+6.2f}m | Y: {filtered_pos[1]:+6.2f}m | "
                      f"Z: {filtered_pos[2]:5.2f}m")

            if mod == "INIŞ_ALGORITMASI" and true_pos[2] < 0.12:
                print(f"\n\n🏁 Helion şehir navigasyonunu bitirdi ve başarıyla hedefe indi!")
                break 
                
    except Exception as e:
        print(f"\nUçuş esnasında hata: {e}")
        import traceback
        traceback.print_exc()

cv2.destroyAllWindows()
print("\n✅ Şehir içi navigasyon ve iniş demosu tamamlandı (v23).")
