import genesis as gs
import numpy as np
import cv2
import time

# ============================================================================
#  drive_v22.py — Şehir İçi ArUco Navigasyon (Düzeltilmiş)
#
#  Akış:  KALKIŞ → ARAMA → HIZALANMA → POSHOLD → DÖNÜŞ (gerekiyorsa) → ARAMA
#         … son marker'da → INIŞ_ALGORITMASI → YER
#
#  v21'deki sorunlar:
#   1) ARAMA/HIZALANMA modlarında yaw_offset=0 → drone kontrolsüz dönüyordu
#   2) Kamera gürültüsü çok yüksekti → ArUco tespiti zorlaşıyordu
#   3) Marker kaybedildiğinde recovery yoktu
#   4) POSHOLD süresinde detected_id değişebiliyordu (son frame'deki ID)
# ============================================================================

def quat_to_rot_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def get_yaw_from_quat(q):
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

# --- 1. DONANIM PARAMETRELERİ ---
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

# --- 4. SİMÜLASYON SAHNE KURULUMU ---
gs.init(backend=gs.cpu)
scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=(0.9, 0.9, 0.9)))

print("🏙️ Şehir ortamı ve ArUco markerları yerleştiriliyor...")

# Marker Planı:
#   Drone (0,0)'dan kalkar, X+ yönünde ilerler
#   (3, 0) → ID 2 = Sola dön → Y+ yönüne geçer
#   (3, 3) → ID 0 = Sağa dön → X+ yönüne geçer
#   (6, 3) → ID 1 = İniş
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

# --- 5. MOTOR VE İTKİ MODELİ ---
class SimulatedMSP:
    def __init__(self):
        self.d_x = 0.15
        self.d_y = 0.15
        self.c_q = 0.04
        self.k_p_angle = 120.0
        self.k_d_angle = 30.0

    def apply_rc_to_physics(self, roll, pitch, yaw, throttle, quat, velocity, true_z, elapsed):
        throttle_pct = max(0.0, (throttle - 1000.0) / 1000.0)
        voltage_drop = (throttle_pct ** 2) * 2.5
        current_voltage = BATTERY_VOLTAGE_MAX - voltage_drop
        dynamic_max_rpm = current_voltage * MOTOR_KV

        base_rpm = throttle_pct * dynamic_max_rpm
        total_thrust = C_T * (base_rpm ** 2)

        # Yer etkisi
        if 0.0 < true_z < 0.4:
            total_thrust *= 1.0 + (0.4 - true_z) * 0.6

        base_thrust = total_thrust / 4.0

        # Hedef açılar
        max_angle = np.deg2rad(25.0)
        target_pitch = ((1500.0 - pitch) / 500.0) * max_angle
        target_roll = ((1500.0 - roll) / 500.0) * max_angle
        target_yaw_rate = ((1500.0 - yaw) / 500.0) * 4.0

        R_mat = quat_to_rot_matrix(quat)
        current_pitch = np.arcsin(np.clip(-R_mat[2, 0], -1, 1))
        current_roll = np.arctan2(R_mat[2, 1], R_mat[2, 2])

        ang_vel = velocity[3:] if len(velocity) == 6 else np.zeros(3)

        pitch_error = target_pitch - current_pitch
        roll_error = target_roll - current_roll
        yaw_rate_error = target_yaw_rate - ang_vel[2]

        pitch_inf = (pitch_error * self.k_p_angle - ang_vel[1] * self.k_d_angle) * base_thrust
        roll_inf = (roll_error * self.k_p_angle - ang_vel[0] * self.k_d_angle) * base_thrust
        yaw_inf = (yaw_rate_error * 25.0) * base_thrust

        max_inf = base_thrust * 0.8
        pitch_inf = np.clip(pitch_inf, -max_inf, max_inf)
        roll_inf = np.clip(roll_inf, -max_inf, max_inf)
        yaw_inf = np.clip(yaw_inf, -max_inf, max_inf)

        T_FL = max(0.0, base_thrust - pitch_inf + roll_inf - yaw_inf)
        T_FR = max(0.0, base_thrust - pitch_inf - roll_inf + yaw_inf)
        T_BL = max(0.0, base_thrust + pitch_inf + roll_inf + yaw_inf)
        T_BR = max(0.0, base_thrust + pitch_inf - roll_inf - yaw_inf)

        F_z = T_FL + T_FR + T_BL + T_BR
        tau_x = (T_BL + T_FL - T_FR - T_BR) * self.d_y
        tau_y = (T_BL + T_BR - T_FL - T_FR) * self.d_x
        tau_z = (T_FR + T_BL - T_FL - T_BR) * self.c_q

        local_force = np.array([0.0, 0.0, F_z])
        local_torque = np.array([tau_x, tau_y, tau_z])

        global_force = R_mat.dot(local_force)
        global_torque = R_mat.dot(local_torque)


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

# --- 7. UÇUŞ KONTROLCÜSÜ ---
class AutonomousFlightController:
    def __init__(self):
        self.pid_alt = PIDController(15.0, 2.5, 40.0, -100.0, 100.0)
        self.pid_pitch = PIDController(60.0, 15.0, 80.0, -200.0, 200.0)
        self.pid_roll = PIDController(60.0, 15.0, 80.0, -200.0, 200.0)
        self.pid_yaw = PIDController(120.0, 5.0, 40.0, -400.0, 400.0)

    def compute_rc_channels(self, current_pos, target_pos, current_yaw, target_yaw, dt, mod):
        self.pid_alt.setpoint = target_pos[2]
        base_throttle = HOVER_PWM if target_pos[2] > 0.3 else 1345.0

        if mod == "INIS" and current_pos[2] < 0.5:
            base_throttle = 1315.0
            self.pid_alt.kp = 6.0
            self.pid_alt.kd = 10.0
        else:
            self.pid_alt.reset_gains()

        alt_offset = self.pid_alt.compute(current_pos[2], dt)

        # Body-frame hata hesabı
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

        # *** KRİTİK DÜZELTME: Yaw HER MODDA stabilize edilir ***
        yaw_error = target_yaw - current_yaw
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
        self.pid_yaw.setpoint = current_yaw + yaw_error
        yaw_offset = self.pid_yaw.compute(current_yaw, dt)

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

mod = "KALKIS"
target = [0.0, 0.0, TARGET_ALTITUDE]
target_yaw = 0.0

# Navigasyon
current_direction = "X+"
last_processed_id = -1
poshold_start = 0.0
poshold_marker_id = -1
align_start_time = 0.0
marker_lost_count = 0

# ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Yön değiştirme fonksiyonları
def turn_left(direction):
    mapping = {"X+": "Y+", "Y+": "X-", "X-": "Y-", "Y-": "X+"}
    return mapping[direction]

def turn_right(direction):
    mapping = {"X+": "Y-", "Y-": "X-", "X-": "Y+", "Y+": "X+"}
    return mapping[direction]

with open("helion_v22_city_log.txt", "w") as log_file:
    print("\n🚀 Helion v22 — Şehir İçi ArUco Navigasyon Başlıyor!")

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

            # Sinematik kamera
            camera.set_pose(
                pos=np.array([true_pos[0] - 1.0, true_pos[1] - 3.0, 3.0]),
                lookat=true_pos
            )

            # Drone alt kamerası (yaw ile döner)
            cam_up = np.array([np.cos(current_yaw), np.sin(current_yaw), 0.0])
            drone_cam.set_pose(
                pos=true_pos,
                lookat=true_pos + np.array([0, 0, -1]),
                up=cam_up
            )

            scene.step()

            # --- ArUco Tespiti (her 3 frame'de bir) ---
            marker_found = False
            detection_attempted = False
            cy_px, cx_px = 240, 320
            detected_id = -1

            if i % 3 == 0:
                detection_attempted = True
                res = drone_cam.render()
                img_rgb = res[0]
                if img_rgb.dtype == np.float32:
                    img_rgb = (img_rgb * 255).astype(np.uint8)

                # Hafif gürültü
                noise = np.zeros(img_rgb.shape, np.uint8)
                cv2.randn(noise, 0, 5)
                img_rgb = cv2.add(img_rgb, noise)

                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                corners, ids, _ = detector.detectMarkers(img_bgr)

                if ids is not None:
                    marker_found = True
                    detected_id = ids[0][0]

                    c = corners[0][0]
                    cx_px = int(np.mean(c[:, 0]))
                    cy_px = int(np.mean(c[:, 1]))

                    cv2.aruco.drawDetectedMarkers(img_bgr, corners, ids)
                    cv2.imwrite("aruco_city_debug.jpg", img_bgr)

                cv2.imshow("Drone Kamera", img_bgr)
                cv2.waitKey(1)

            # ==================================================================
            #  DURUM MAKİNESİ
            # ==================================================================

            if mod == "KALKIS":
                if true_pos[2] > TARGET_ALTITUDE - 0.2:
                    mod = "ARAMA"
                    print(f"\n🎯 {TARGET_ALTITUDE}m'ye çıkıldı. {current_direction} yönünde arama başlıyor...")

            elif mod == "ARAMA":
                # Mevcut pozisyonun biraz ilerisine hedef koy (lineer ilerleme)
                lead_dist = 0.09  # Hedef her zaman 8cm ileride
                if current_direction == "X+":
                    target[0] = filtered_pos[0] + lead_dist
                    target[1] = filtered_pos[1]  # Y sabit
                elif current_direction == "X-":
                    target[0] = filtered_pos[0] - lead_dist
                    target[1] = filtered_pos[1]
                elif current_direction == "Y+":
                    target[0] = filtered_pos[0]  # X sabit
                    target[1] = filtered_pos[1] + lead_dist
                elif current_direction == "Y-":
                    target[0] = filtered_pos[0]
                    target[1] = filtered_pos[1] - lead_dist

                target[2] = TARGET_ALTITUDE

                # Marker bulundu ve daha önce işlenmemiş bir ID ise
                if marker_found and detected_id != last_processed_id:
                    print(f"\n👀 [{elapsed:.1f}s] ArUco ID {detected_id} görüldü! Hizalanıyor...")
                    # Mevcut pozisyonda dur
                    target[0] = filtered_pos[0]
                    target[1] = filtered_pos[1]
                    mod = "HIZALANMA"
                    align_start_time = elapsed
                    marker_lost_count = 0

            elif mod == "HIZALANMA":
                target[2] = TARGET_ALTITUDE

                # Sadece tespit yapılan frame'lerde marker mantığını işle
                if detection_attempted:
                    if marker_found:
                        marker_lost_count = 0
                        error_px = cx_px - 320
                        error_py = 240 - cy_px

                        # Drone yaw≈0 olduğu için body ≈ global
                        target[0] = filtered_pos[0] + error_py * 0.001
                        target[1] = filtered_pos[1] - error_px * 0.001

                        if abs(error_px) < 50 and abs(error_py) < 50:
                            print(f"✅ Marker ID {detected_id} ortalandı! POSHOLD'a geçiliyor...")
                            target[0] = filtered_pos[0]
                            target[1] = filtered_pos[1]
                            mod = "POSHOLD"
                            poshold_start = elapsed
                            poshold_marker_id = detected_id  # ID'yi kilitle
                    else:
                        marker_lost_count += 1
                        # 50 tespit denemesinde (~2.5s) marker bulunamazsa aramaya dön
                        if marker_lost_count > 50:
                            print(f"⚠️ Marker kayboldu ({marker_lost_count} miss), ARAMA'ya dönülüyor...")
                            mod = "ARAMA"

                # Hizalanma zaman aşımı (10 saniye)
                if elapsed - align_start_time > 10.0:
                    print(f"⚠️ Hizalanma zaman aşımı, ARAMA'ya dönülüyor...")
                    mod = "ARAMA"

            elif mod == "POSHOLD":
                target[2] = TARGET_ALTITUDE

                # 2 saniye bekle, sonra kilitlenen ID'ye göre hareket et
                if elapsed - poshold_start > 2.0:
                    mid = poshold_marker_id

                    if mid == 2:  # Sola Dön
                        new_dir = turn_left(current_direction)
                        print(f"\n↩️ Sola dönülüyor: {current_direction} → {new_dir}")
                        current_direction = new_dir
                        mod = "ARAMA"

                    elif mid == 0:  # Sağa Dön
                        new_dir = turn_right(current_direction)
                        print(f"\n↪️ Sağa dönülüyor: {current_direction} → {new_dir}")
                        current_direction = new_dir
                        mod = "ARAMA"

                    elif mid == 3:  # İleri Git
                        print(f"\n⬆️ İleri devam ediliyor.")
                        mod = "ARAMA"

                    elif mid == 1:  # İniş
                        print(f"\n🔽 İniş başlatılıyor!")
                        mod = "INIS"

                    last_processed_id = mid

            elif mod == "INIS":
                target[2] = 0.05

                if marker_found and detected_id == 1:
                    epx = cx_px - 320
                    epy = 240 - cy_px
                    # Drone yaw≈0 olduğu için body ≈ global
                    target[0] = filtered_pos[0] + epy * 0.002
                    target[1] = filtered_pos[1] - epx * 0.002

            # --- Kontrol Hesapla ve Uygula ---
            roll_pwm, pitch_pwm, yaw_pwm, throttle_pwm = fc.compute_rc_channels(
                filtered_pos, target, current_yaw, target_yaw, dt, mod
            )
            current_voltage = msp_bridge.apply_rc_to_physics(
                roll_pwm, pitch_pwm, yaw_pwm, throttle_pwm,
                true_quat, true_vel, true_pos[2], elapsed
            )

            # --- Log ---
            yaw_rate_log = abs(true_vel[5]) if len(true_vel) >= 6 else 0.0
            if i % 40 == 0:
                print(f"⏱️ {elapsed:05.1f}s | Mod: {mod:15} | Yön: {current_direction} | "
                      f"Yaw: {np.degrees(current_yaw):+6.0f}° | YR: {yaw_rate_log:.1f} | "
                      f"X: {filtered_pos[0]:+6.2f}m | Y: {filtered_pos[1]:+6.2f}m | "
                      f"Z: {filtered_pos[2]:5.2f}m")

            # İniş tamamlandı mı?
            if mod == "INIS" and true_pos[2] < 0.12:
                print(f"\n\n🏁 Helion şehir navigasyonunu bitirdi ve başarıyla hedefe indi!")
                break

    except Exception as e:
        print(f"\nUçuş esnasında hata: {e}")
        import traceback
        traceback.print_exc()

cv2.destroyAllWindows()
print("\n✅ Şehir içi navigasyon ve iniş demosu tamamlandı.")
