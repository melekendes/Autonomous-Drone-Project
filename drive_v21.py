import genesis as gs
import numpy as np
import cv2
import time
import random

def quat_to_rot_matrix(q):
    # q is [w, x, y, z]
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def get_yaw_from_quat(q):
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# --- 1. DONANIM VE FİZİKSEL PARAMETRELER ---
DRONE_MASS = 18.11 / 9.81 
GRAVITY = 9.81
HOVER_PWM = 1362.0
CENTER_PWM = 1500.0

BATTERY_VOLTAGE_MAX = 22.2  
MOTOR_KV = 900.0        
MAX_RPM = BATTERY_VOLTAGE_MAX * MOTOR_KV  

C_T = 3.4618e-7
TARGET_ALTITUDE = 1.5       

# --- 2. SENSÖR SİMÜLATÖRÜ ---
class SimulatedSensors:
    def __init__(self):
        self.gps_noise_std = 0.02  
        self.baro_noise_std = 0.01 
        self.z_drift = 0.0         

    def read_sensors(self, true_pos, dt):
        self.z_drift += np.random.normal(0, 0.0005) * dt 
        noisy_x = true_pos[0] + np.random.normal(0, self.gps_noise_std)
        noisy_y = true_pos[1] + np.random.normal(0, self.baro_noise_std)
        noisy_z = true_pos[2] + np.random.normal(0, self.baro_noise_std) + self.z_drift
        return np.array([noisy_x, noisy_y, noisy_z])

# --- 3. ALÇAK GEÇİREN FİLTRE ---
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

# --- 4. SİMÜLASYON VE SAHNE KURULUMU (ŞEHİR VE ARUCO YOLU) ---
gs.init(backend=gs.cpu)
scene = gs.Scene(show_viewer=True)
# Arka planı daha açık renkli göstermek için zemin materyalini kullanıyoruz
scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(0.9, 0.9, 0.9)))

print("🏙️ Şehir ortamı (Binalar) ve ArUco markerları yerleştiriliyor...")

# Markerlar (Yol Tarifi)
# Drone (0,0)'dan başlayıp +X yönünde gidiyor.


# 3 metrede Sola Dönüş (ID 2)
scene.add_entity(gs.morphs.URDF(file='aruco_turn_left.urdf', pos=(3.0, 0.0, 0.0)))

# Drone sola dönüp +Y yönünde gidiyor.
# Y ekseninde 3 metrede Sağa Dönüş (ID 0)
scene.add_entity(gs.morphs.URDF(file='aruco_turn_right.urdf', pos=(3.0, 3.0, 0.0)))

# Drone sağa dönüp +X yönünde gidiyor.
# X ekseninde 6 metrede İniş (ID 1)
scene.add_entity(gs.morphs.URDF(file='aruco_land.urdf', pos=(6.0, 3.0, 0.0)))

# Binalar (Engeller)
# Yol boyunca kenarlara binalar yerleştiriyoruz ki bir şehir sokağı hissi versin.
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

# --- 5. DONANIMSAL MOTOR VE VEKTÖREL İTKİ MODELİ ---
class SimulatedMSP:
    def __init__(self):
        # 4 Motor konfigürasyonu için merkez uzaklıkları
        self.d_x = 0.16
        self.d_y = 0.11
        self.c_q = 0.015  # Tork/İtki oranı
        
        # İç döngü (Betaflight Angle Modu) PID katsayıları
        self.k_p_angle = 120.0
        self.k_d_angle = 30.0
        
    def apply_rc_to_physics(self, roll, pitch, yaw, throttle, quat, velocity, true_z, elapsed):
        throttle_pct = max(0.0, (throttle - 1000.0) / 1000.0)
        voltage_drop = (throttle_pct ** 2) * 2.5 
        current_voltage = BATTERY_VOLTAGE_MAX - voltage_drop
        dynamic_max_rpm = current_voltage * MOTOR_KV
        
        base_rpm = throttle_pct * dynamic_max_rpm
        total_thrust = C_T * (base_rpm ** 2) 

        # Yer etkisi (Ground effect)
        if true_z < 0.4 and true_z > 0.0:
            ground_cushion_multiplier = 1.0 + (0.4 - true_z) * 0.6 
            total_thrust *= ground_cushion_multiplier

        # 4 motora bölüyoruz
        base_thrust = total_thrust / 4.0

        # Betaflight Angle Modu Simülasyonu
        # RC komutlarını (1000-2000) hedef açılara (radyan) çevir
        max_angle = np.deg2rad(25.0)
        target_pitch = ((1500.0 - pitch) / 500.0) * max_angle
        target_roll = ((1500.0 - roll) / 500.0) * max_angle
        target_yaw_rate = ((1500.0 - yaw) / 500.0) * 2.0  # rad/s

        # Mevcut açıları bul (Euler XYZ)
        # scipy kullanmadan:
        R_mat = quat_to_rot_matrix(quat)
        current_pitch = np.arcsin(-R_mat[2, 0])
        current_roll = np.arctan2(R_mat[2, 1], R_mat[2, 2])
        
        ang_vel = velocity[3:] if len(velocity) == 6 else np.zeros(3)
        
        # İç döngü P/D kontrolcüsü (Tork üretimi için)
        pitch_error = target_pitch - current_pitch
        roll_error = target_roll - current_roll
        yaw_rate_error = target_yaw_rate - ang_vel[2]

        pitch_inf = (pitch_error * self.k_p_angle - ang_vel[1] * self.k_d_angle) * base_thrust
        roll_inf = (roll_error * self.k_p_angle - ang_vel[0] * self.k_d_angle) * base_thrust
        yaw_inf = (yaw_rate_error * 10.0) * base_thrust
        
        # Motorları limitle
        max_inf = base_thrust * 0.8
        pitch_inf = np.clip(pitch_inf, -max_inf, max_inf)
        roll_inf = np.clip(roll_inf, -max_inf, max_inf)
        yaw_inf = np.clip(yaw_inf, -max_inf, max_inf)

        # FL (CW), FR (CCW), BL (CCW), BR (CW)
        T_FL = max(0.0, base_thrust - pitch_inf + roll_inf - yaw_inf)
        T_FR = max(0.0, base_thrust - pitch_inf - roll_inf + yaw_inf)
        T_BL = max(0.0, base_thrust + pitch_inf + roll_inf + yaw_inf)
        T_BR = max(0.0, base_thrust + pitch_inf - roll_inf - yaw_inf)

        # Lokal kuvvet ve tork hesaplamaları
        F_z = T_FL + T_FR + T_BL + T_BR
        tau_x = (T_BL + T_FL - T_FR - T_BR) * self.d_y
        tau_y = (T_BL + T_BR - T_FL - T_FR) * self.d_x
        tau_z = (T_FR + T_BL - T_FL - T_BR) * self.c_q

        # Global eksene çevirme (Gerçekçi Rotasyon Dinamiği)
        local_force = np.array([0.0, 0.0, F_z])
        local_torque = np.array([tau_x, tau_y, tau_z])
        
        global_force = R_mat.dot(local_force)
        global_torque = R_mat.dot(local_torque)
        
        # Hava direnci ve Drag modellemesi
        if len(velocity) == 6:
            lin_vel = velocity[:3]
            c_d = 0.6    # Lineer drag katsayısı
            c_ad = 0.08  # Açısal drag katsayısı
            global_force -= c_d * lin_vel
            global_torque -= c_ad * ang_vel



        kuvvet = np.zeros(6)
        kuvvet[0:3] = global_force
        kuvvet[3:6] = global_torque
        drone.control_dofs_force(kuvvet)
        
        return current_voltage

# --- 6. PID KONTROLCÜ ---
class PIDController:
    def __init__(self, kp, ki, kd, out_min, out_max):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint, self.integral, self.last_error = 0.0, 0.0, 0.0
        self.out_min, self.out_max = out_min, out_max
        self.orig_kp, self.orig_kd = kp, kd 

    def compute(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral = np.clip(self.integral + error * dt, -500.0, 500.0)
        p_out = self.kp * error
        i_out = self.ki * self.integral
        d_out = self.kd * (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error
        return np.clip(p_out + i_out + d_out, self.out_min, self.out_max)
        
    def reset_gains(self):
        self.kp, self.kd = self.orig_kp, self.orig_kd

# --- 7. OTONOM UÇUŞ KONTROLCÜSÜ ---
class AutonomousFlightController:
    def __init__(self):
        self.pid_alt = PIDController(15.0, 2.5, 40.0, -100.0, 100.0)
        self.pid_pitch = PIDController(60.0, 15.0, 80.0, -200.0, 200.0) 
        self.pid_roll = PIDController(60.0, 15.0, 80.0, -200.0, 200.0)  
        self.pid_yaw = PIDController(40.0, 2.0, 30.0, -200.0, 200.0)

    def compute_rc_channels(self, current_pos, target_pos, current_yaw, target_yaw, dt, mod):
        self.pid_alt.setpoint = target_pos[2]
        base_throttle = HOVER_PWM if target_pos[2] > 0.3 else 1345.0 
        
        if mod == "INIŞ_ALGORITMASI" and current_pos[2] < 0.5:
            base_throttle = 1315.0 
            self.pid_alt.kp = 6.0  
            self.pid_alt.kd = 10.0 
        else:
            self.pid_alt.reset_gains()

        alt_offset = self.pid_alt.compute(current_pos[2], dt)

        # Global hatayı drone'un body eksenine çevir
        error_global_x = target_pos[0] - current_pos[0]
        error_global_y = target_pos[1] - current_pos[1]

        cy = np.cos(-current_yaw)
        sy = np.sin(-current_yaw)
        error_body_x = error_global_x * cy - error_global_y * sy
        error_body_y = error_global_x * sy + error_global_y * cy

        self.pid_pitch.setpoint = error_body_x
        self.pid_roll.setpoint = error_body_y
        
        pitch_offset = self.pid_pitch.compute(0.0, dt)
        roll_offset = self.pid_roll.compute(0.0, dt)

        # Yaw sadece DONUS modunda aktif, diğer modlarda nötr
        if mod == "DONUS":
            yaw_error = target_yaw - current_yaw
            yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
            self.pid_yaw.setpoint = current_yaw + yaw_error
            yaw_offset = self.pid_yaw.compute(current_yaw, dt)
        else:
            yaw_offset = 0.0
            self.pid_yaw.integral = 0.0
            self.pid_yaw.last_error = 0.0

        final_throttle = np.clip(base_throttle + alt_offset, 1000.0, 2000.0)
        final_pitch = np.clip(CENTER_PWM - pitch_offset, 1000.0, 2000.0)
        final_roll = np.clip(CENTER_PWM + roll_offset, 1000.0, 2000.0)
        final_yaw = np.clip(CENTER_PWM - yaw_offset, 1000.0, 2000.0)
        
        return final_roll, final_pitch, final_yaw, final_throttle

# --- 8. ANA UÇUŞ DÖNGÜSÜ ---
sensors = SimulatedSensors()
pos_filter = VectorLowPassFilter(alpha=0.15) 
msp_bridge = SimulatedMSP()
fc = AutonomousFlightController()

last_log_time = 0.0

mod = "KALKIS"
target = [0.0, 0.0, TARGET_ALTITUDE]
target_yaw = 0.0

# Navigasyon Değişkenleri
current_direction = "X+" # İlk hareket yönü
last_turn_time = -15.0 # Markerı birden fazla kez okumaması için cooldown
last_processed_id = -1
poshold_start = 0.0

# OpenCV ArUco ayarlari
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

with open("helion_v21_city_log.txt", "w") as log_file:
    print("\n🚀 Helion 10 v21 - Şehir İçi ArUco Navigasyon Demosu Başlıyor!")
    
    try:
        for i in range(15000): 
            dt = 0.015
            elapsed = i * dt
            
            true_pos = drone.get_pos().numpy()
            true_quat = drone.get_quat().numpy()
            true_vel = drone.get_dofs_velocity().numpy()
            
            noisy_pos = sensors.read_sensors(true_pos, dt)
            filtered_pos = pos_filter.compute(noisy_pos)
            current_yaw = get_yaw_from_quat(true_quat)
            
            # Sinematik takip kamerası
            camera.set_pose(pos=np.array([true_pos[0]-1.0, true_pos[1]-3.0, 3.0]), lookat=true_pos)
            
            # Drone alt kamerası (Drone'un yaw açısıyla birlikte döner)
            cam_up = np.array([np.cos(current_yaw), np.sin(current_yaw), 0.0])
            drone_cam.set_pose(pos=true_pos, lookat=true_pos + np.array([0, 0, -1]), up=cam_up)
            
            scene.step()

            marker_found = False
            cy, cx = 240, 320
            detected_id = -1
            
            if i % 3 == 0:
                res = drone_cam.render()
                img_rgb = res[0]
                if img_rgb.dtype == np.float32:
                    img_rgb = (img_rgb * 255).astype(np.uint8)
                
                # Gerçekçi kamera gürültüsü (Gaussian Noise) ekleniyor
                noise = np.zeros(img_rgb.shape, np.uint8)
                cv2.randn(noise, 0, 15)  # mean=0, std=15
                img_rgb = cv2.add(img_rgb, noise)
                
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                corners, ids, rejected = detector.detectMarkers(img_bgr)
                
                if ids is not None:
                    marker_found = True
                    detected_id = ids[0][0] # Görülen ilk marker ID'si
                    
                    c = corners[0][0]
                    cx = int(np.mean(c[:, 0]))
                    cy = int(np.mean(c[:, 1]))
                    
                    cv2.aruco.drawDetectedMarkers(img_bgr, corners, ids)
                    cv2.imwrite("aruco_city_debug.jpg", img_bgr)
                    
                # Ayrı pencerede göster
                cv2.imshow("Drone Kamera", img_bgr)
                cv2.waitKey(1)

            # --- DİNAMİK DURUM MAKİNESİ (STATE MACHINE) ---
            if mod == "KALKIS":
                if true_pos[2] > TARGET_ALTITUDE - 0.2:
                    mod = "ARAMA"
                    print(f"\n🎯 {TARGET_ALTITUDE}m'ye çıkıldı. Şehir rotasında ilerleme başlıyor...")
                    
            elif mod == "ARAMA":
                # Belirlenen yönde hareket et
                move_speed = 0.003
                if current_direction == "X+": 
                    target[0] += move_speed
                    target_yaw = 0.0
                elif current_direction == "X-": 
                    target[0] -= move_speed
                    target_yaw = np.pi
                elif current_direction == "Y+": 
                    target[1] += move_speed
                    target_yaw = np.pi / 2.0
                elif current_direction == "Y-": 
                    target[1] -= move_speed
                    target_yaw = -np.pi / 2.0
                
                target[2] = TARGET_ALTITUDE
                
                if marker_found and (elapsed - last_turn_time) > 15.0 and detected_id != last_processed_id:
                    print(f"\n👀 [Zaman: {elapsed:.1f}s] ArUco ID {detected_id} okundu! Merkeze hizalanıyor...")
                    target[0] = filtered_pos[0]
                    target[1] = filtered_pos[1]
                    mod = "HIZALANMA"
                    
            elif mod == "HIZALANMA":
                target[2] = TARGET_ALTITUDE
                if marker_found:
                    error_px = cx - 320
                    error_py = 240 - cy
                    
                    # Confirmed by debug: right in image = -Y world, below in image = -X world
                    err_body_fwd = error_py * 0.001
                    err_body_right = -error_px * 0.001
                    
                    error_global_x = err_body_fwd * np.cos(current_yaw) - err_body_right * np.sin(current_yaw)
                    error_global_y = err_body_fwd * np.sin(current_yaw) + err_body_right * np.cos(current_yaw)
                    
                    target[0] = filtered_pos[0] + error_global_x
                    target[1] = filtered_pos[1] + error_global_y
                    
                    if abs(error_px) < 50 and abs(error_py) < 50:
                        print(f"\n✅ Marker ortalandı! Pozisyon sabitleniyor...")
                        target[0] = filtered_pos[0]
                        target[1] = filtered_pos[1]
                        mod = "POSHOLD"
                        poshold_start = elapsed

            elif mod == "POSHOLD":
                # Pozisyonu koru, 2 saniye bekle
                target[2] = TARGET_ALTITUDE
                if elapsed - poshold_start > 2.0:
                    if detected_id == 0: # Sağa Dön
                        print(f"\n↪️ Sağa dönülüyor.")
                        if current_direction == "X+": 
                            current_direction = "Y-"
                            target_yaw = -np.pi / 2.0
                        elif current_direction == "Y-": 
                            current_direction = "X-"
                            target_yaw = np.pi
                        elif current_direction == "X-": 
                            current_direction = "Y+"
                            target_yaw = np.pi / 2.0
                        elif current_direction == "Y+": 
                            current_direction = "X+"
                            target_yaw = 0.0
                        mod = "DONUS"
                                
                    elif detected_id == 2: # Sola Dön
                        print(f"\n↩️ Sola dönülüyor.")
                        if current_direction == "X+": 
                            current_direction = "Y+"
                            target_yaw = np.pi / 2.0
                        elif current_direction == "Y+": 
                            current_direction = "X-"
                            target_yaw = np.pi
                        elif current_direction == "X-": 
                            current_direction = "Y-"
                            target_yaw = -np.pi / 2.0
                        elif current_direction == "Y-": 
                            current_direction = "X+"
                            target_yaw = 0.0
                        mod = "DONUS"

                    elif detected_id == 3: # İleri Git
                        print(f"\n⬆️ İleri devam ediliyor.")
                        mod = "ARAMA"
                            
                    elif detected_id == 1: # İniş
                        print(f"\n🔽 İniş başlatılıyor.")
                        mod = "INIŞ_ALGORITMASI"
                    
                    last_processed_id = detected_id
                    last_turn_time = elapsed

            elif mod == "DONUS":
                # Dönüş sırasında konumunu koru (Hover)
                target[2] = TARGET_ALTITUDE
                
                # Yaw hatasını hesapla
                yaw_diff = abs((target_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi)
                
                # Eğer hedefe (yaklaşık 5 derece toleransla) ulaştıysa ARAMA moduna dön
                if yaw_diff < 0.1:
                    print(f"\n✅ Dönüş tamamlandı. {current_direction} yönünde devam ediliyor.")
                    mod = "ARAMA"

            elif mod == "INIŞ_ALGORITMASI":
                target[2] = 0.05
                if marker_found and detected_id == 1:
                    epx = cx - 320
                    epy = 240 - cy
                    efwd = epy * 0.002
                    eright = -epx * 0.002
                    egx = efwd * np.cos(current_yaw) - eright * np.sin(current_yaw)
                    egy = efwd * np.sin(current_yaw) + eright * np.cos(current_yaw)
                    target[0] = filtered_pos[0] + egx
                    target[1] = filtered_pos[1] + egy
            
            roll_pwm, pitch_pwm, yaw_pwm, throttle_pwm = fc.compute_rc_channels(filtered_pos, target, current_yaw, target_yaw, dt, mod)
            current_voltage = msp_bridge.apply_rc_to_physics(roll_pwm, pitch_pwm, yaw_pwm, throttle_pwm, true_quat, true_vel, true_pos[2], elapsed)
            
            if i % 40 == 0:
                print(f"⏱️ {elapsed:04.1f}s | Mod: {mod:15} | Yön: {current_direction} | Yaw: {np.degrees(current_yaw):.0f}° | X: {filtered_pos[0]:.2f}m | Y: {filtered_pos[1]:.2f}m | Z: {filtered_pos[2]:.2f}m")

            if mod == "INIŞ_ALGORITMASI" and true_pos[2] < 0.12:
                print(f"\n\n🏁 Helion şehir navigasyonunu bitirdi ve başarıyla hedefe indi!")
                break 
                
    except Exception as e:
        print(f"\nUçuş esnasında hata: {e}")
        import traceback
        traceback.print_exc()

print("\n✅ Şehir içi navigasyon ve iniş demosu başarıyla tamamlandı.")
