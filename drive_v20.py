import genesis as gs
import numpy as np
import cv2
import time
import random

# --- 1. DONANIM VE FİZİKSEL PARAMETRELER (REAL WORLD SPEC) ---
DRONE_MASS = 18.11 / 9.81 
GRAVITY = 9.81
HOVER_PWM = 1362.0
CENTER_PWM = 1500.0

BATTERY_VOLTAGE_MAX = 22.2  # 6S Tam Dolu LiPo Batarya
MOTOR_KV = 900.0        
MAX_RPM = BATTERY_VOLTAGE_MAX * MOTOR_KV  

C_T = 3.4618e-7

TARGET_ALTITUDE = 1.5       # 🚀 Kalkış Yükseklik Hedefi (1.5 Metre)

MIN_RC_VALUE = 1400            
MAX_RC_VALUE = 1600            

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

# --- 4. SİMÜLASYON VE SAHNE KURULUMU ---
gs.init(backend=gs.cpu)
scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane()) 

# ArUco marker'ı rastgele bir X konumuna yerleştiriyoruz (aynı eksende aranacak)
aruco_x = random.uniform(2.0, 5.0)
print(f"📍 İniş Noktası (ArUco Marker) X = {aruco_x:.2f} metre konumuna yerleştirildi.")
marker = scene.add_entity(gs.morphs.URDF(file='aruco_land.urdf', pos=(aruco_x, 0.0, 0.0)))

drone = scene.add_entity(gs.morphs.URDF(file='helionv4.urdf', pos=(0.0, 0.0, 0.05)))

# Kamerayı drone'un ilerleyişini geniş açıdan görecek şekilde konumlandırıyoruz
camera = scene.add_camera(res=(640, 480), pos=(3.5, -3.5, 2.0), lookat=(1.0, 0.0, 0.8), fov=65)

# Drone'un altındaki, aşağıya bakan FPV kamerası (ArUco tespiti için)
drone_cam = scene.add_camera(res=(640, 480), pos=(0.0, 0.0, 1.0), lookat=(0.0, 0.0, 0.0), fov=65)

scene.build()

# --- 5. DONANIMSAL MOTOR VE VEKTÖREL İTKİ MODELİ ---
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

# --- 6. PID KONTROLCÜ ---
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

# --- 7. OTONOM UÇUŞ KONTROLCÜSÜ ---
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

# --- 8. ANA UÇUŞ DÖNGÜSÜ ---
sensors = SimulatedSensors()
pos_filter = VectorLowPassFilter(alpha=0.15) 
msp_bridge = SimulatedMSP()
fc = AutonomousFlightController()

last_log_time = 0.0

mod = "KALKIS"
target = [0.0, 0.0, TARGET_ALTITUDE]

# OpenCV ArUco ayarlari
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

with open("helion_v20_demo_log.txt", "w") as log_file:
    log_file.write(f"{'Zaman(s)':<10}\t{'Ucus_Modu':<15}\t{'Gercek_X':<10}\t{'Gercek_Z':<10}\t{'Filtreli_Z':<12}\t{'Roll_PWM':<10}\t{'Pitch_PWM':<10}\t{'Throt_PWM':<10}\t{'Voltaj(V)':<10}\n")
    log_file.write("-" * 115 + "\n")
    
    print("\n🚀 Helion 10 v20 - ArUco Arama ve İniş Demosu Başlıyor!")
    
    try:
        for i in range(5000): 
            dt = 0.015
            elapsed = i * dt
            
            true_pos = drone.get_pos().numpy()
            noisy_pos = sensors.read_sensors(true_pos, dt)
            filtered_pos = pos_filter.compute(noisy_pos)
            
            # Sinematik takip kamerası: Drone'u geniş açıdan odaklar
            camera.set_pose(pos=np.array([true_pos[0]+3.0, -3.0, 2.0]), lookat=true_pos)
            
            # Drone alt kamerası: Tam aşağı doğru (Z ekseninde -1 yönünde) bakar. 
            # up=[1, 0, 0] ile X ekseni görüntüde yukarıya denk gelecek şekilde ayarlanır.
            drone_cam.set_pose(pos=true_pos, lookat=true_pos + np.array([0, 0, -1]), up=np.array([1, 0, 0]))
            
            scene.step()

            # OpenCV ile ArUco Tespiti (Performans için her 3 frame'de bir okuyoruz)
            marker_found = False
            cy = 240
            cx = 320
            
            if i % 3 == 0:
                res = drone_cam.render()
                img_rgb = res[0]
                if img_rgb.dtype == np.float32:
                    img_rgb = (img_rgb * 255).astype(np.uint8)
                
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                corners, ids, rejected = detector.detectMarkers(img_bgr)
                
                if ids is not None and 1 in ids:
                    marker_found = True
                    idx = np.where(ids == 1)[0][0]
                    
                    c = corners[idx][0]
                    cx = int(np.mean(c[:, 0]))
                    cy = int(np.mean(c[:, 1]))
                    
                    cv2.aruco.drawDetectedMarkers(img_bgr, corners, ids)
                    cv2.circle(img_bgr, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.imwrite("aruco_debug.jpg", img_bgr) # Debug için kaydet

            # --- DİNAMİK DURUM MAKİNESİ (STATE MACHINE) ---
            if mod == "KALKIS":
                target = [0.0, 0.0, TARGET_ALTITUDE]
                if abs(filtered_pos[2] - TARGET_ALTITUDE) < 0.05:
                    mod = "ARAMA"
                    print(f"\n🎯 {TARGET_ALTITUDE}m'ye çıkıldı. ArUco Marker aranıyor...")
                    
            elif mod == "ARAMA":
                target[0] += 0.01  # Yavaşça X ekseninde ilerleyerek ara
                target[2] = TARGET_ALTITUDE
                
                if marker_found:
                    mod = "HIZALANMA"
                    print(f"\n👀 ArUco Marker (ID:1) Görüldü! Hizalanıyor... (Görüntü Merkezi Y: {cy})")
                    
            elif mod == "HIZALANMA":
                if marker_found:
                    # Görüntü boyutu 640x480. Merkez Y=240. 
                    error_x = 240 - cy 
                    
                    # Yüksekliğe bağlı yaklaşık pixel-to-meter dönüşümü
                    meters_per_pixel = 0.004 * filtered_pos[2]
                    target[0] = filtered_pos[0] + (error_x * meters_per_pixel)
                    
                    if abs(error_x) < 30: # Merkezden belli bir tolerans içinde
                        mod = "INIŞ_ALGORITMASI"
                        print(f"\n🪂 Marker ortalandı! İniş başlatılıyor. Sabit Hedef X: {target[0]:.2f}m")
                # marker_found == False ise target[0]'ı güncellemiyoruz, son kilitlenen hedefte sabit durmasını sağlıyoruz.

            elif mod == "INIŞ_ALGORITMASI":
                target[2] = 0.05 # Yere doğru in
                
                if marker_found:
                    # İnerken de daha hassas iniş için marker'ı takip et
                    error_x = 240 - cy
                    meters_per_pixel = 0.004 * filtered_pos[2]
                    target[0] = filtered_pos[0] + (error_x * meters_per_pixel)
                # marker görünmüyorsa hedef sabit kalır (sürüklenmeyi engeller)
            
            roll_pwm, pitch_pwm, throttle_pwm = fc.compute_rc_channels(filtered_pos, target, dt, mod)
            current_voltage = msp_bridge.apply_rc_to_physics(roll_pwm, pitch_pwm, throttle_pwm, true_pos[2])
            
            if elapsed - last_log_time >= 1.0:
                log_file.write(f"{elapsed:<10.2f}\t{mod:<15}\t{true_pos[0]:<10.3f}\t{true_pos[2]:<10.3f}\t{filtered_pos[2]:<12.3f}\t{roll_pwm:<10.1f}\t{pitch_pwm:<10.1f}\t{throttle_pwm:<10.1f}\t{current_voltage:<10.2f}\n")
                last_log_time = elapsed
                
            if i % 20 == 0:
                print(f"\r⏱️ {elapsed:04.1f}s | Mod: {mod:<15} | X: {filtered_pos[0]:.2f}m | Z: {filtered_pos[2]:.2f}m", end="")

            if mod == "INIŞ_ALGORITMASI" and true_pos[2] < 0.12:
                print(f"\n\n🏁 Helion başarıyla ArUco Marker (ID:1) üzerine iniş yaptı!")
                break 
                
    except Exception as e:
        print(f"\nUçuş esnasında hata: {e}")
        import traceback
        traceback.print_exc()

print("\n✅ ArUco arama ve iniş demosu başarıyla tamamlandı.")
