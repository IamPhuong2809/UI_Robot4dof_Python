import math
import numpy as np
L1 = 56.35; L2 = 120; L3 = 100; L4 = 82; d1 = 170

def Forward_Kinematics(data):

    theta1 = data[0]*math.pi / 180
    theta2 = data[1] * math.pi / 180
    theta3 = data[2] * math.pi / 180
    theta4 = data[3] * math.pi / 180

    # Tính tọa độ Px
    Px = math.cos(theta1) * ( L1 + L2 * math.cos(theta2) + L3 * math.cos(theta2 + theta3) + L4 * math.cos(theta2 + theta3 + theta4))
    Px = round(Px, 2)

    # Tính tọa độ Py
    Py = math.sin(theta1) * ( L1 + L2 * math.cos(theta2) + L3 * math.cos(theta2 + theta3) + L4 * math.cos(theta2 + theta3 + theta4))
    Py = round(Py, 2)

    # Tính tọa độ Pz
    Pz = (d1 + L3 * math.sin(theta2 + theta3) + L2 * math.sin(theta2) + L4 * math.sin(theta2 + theta3 + theta4))
    Pz = round(Pz, 1)

    return [Px,  Py,   Pz]

def Inverse_Kinematics(data):

    Px = data[0]
    Py = data[1]
    Pz = data[2]
    theta = data[3]
    theta_rad = theta*math.pi/180

    #Theta1
    k = math.sqrt(Px**2 + Py**2)
    theta1_rad = math.atan2(Py / k, Px / k)
    Theta1 = math.degrees(theta1_rad) 
    Theta1 = round(Theta1, 0)

    if Theta1 < -180:
        Theta1 = Theta1 + 360
    elif Theta1 > 180:
        Theta1 = Theta1 - 360

    E = Px * math.cos(theta1_rad) + Py * math.sin(theta1_rad) - L1 - L4 * math.cos(theta_rad)
    F = Pz - d1 - L4 * math.sin(theta_rad)

    a = -2 * L2 * F
    b = -2 * L2 * E
    d = L3**2 - E**2 - F**2 - L2**2
    f = math.sqrt(a**2 + b**2)  # đổi e thành f 
    anpha = math.atan2((-2 * L2 * F) / f, (-2 * L2 * E) / f)

    var_temp = d**2 / f**2
    if var_temp > 1:
        var_temp = 1

    #SET 1------------------------------------------------
    #Theta2
    theta2_1_rad = math.atan2(math.sqrt(1 - var_temp), d / f) + anpha
    Theta2_1 = math.degrees(theta2_1_rad)  # Chuyển sang độ
    Theta2_1 = round(Theta2_1, 0)

    if Theta2_1 < -180:
        Theta2_1 = Theta2_1 + 360
    elif Theta2_1 > 180:
        Theta2_1 = Theta2_1 - 360

    # Theta3
    c23 = (Px * math.cos(theta1_rad) + Py * math.sin(theta1_rad) - L1 - L2 * math.cos(theta2_1_rad) - L4 * math.cos(theta_rad)) / L3
    s23 = (Pz - d1 - L2 * math.sin(theta2_1_rad) - L4 * math.sin(theta_rad)) / L3
    theta3_1_rad = math.atan2(s23, c23) - theta2_1_rad
    Theta3_1 = math.degrees(theta3_1_rad)  # Chuyển sang độ
    Theta3_1 = round(Theta3_1, 0)

    if Theta3_1 < -180:
        Theta3_1 = Theta3_1 + 360
    elif Theta3_1 > 180:
        Theta3_1 = Theta3_1 - 360

    # Theta4
    theta4_1_rad = theta_rad - theta2_1_rad - theta3_1_rad
    Theta4_1 = math.degrees(theta4_1_rad)  # Chuyển sang độ
    Theta4_1 = round(Theta4_1, 0)

    #Set 2-------------------------------------------------------------------------

    theta2_2_rad = math.atan2(-math.sqrt(1 - var_temp), d / f) + anpha
    Theta2_2 = math.degrees(theta2_2_rad)  # Chuyển sang độ
    Theta2_2 = round(Theta2_2, 0)

    if Theta2_2 < -180:
        Theta2_2 = Theta2_2 + 360
    elif Theta2_2 > 180:
        Theta2_2 = Theta2_2 - 360

    # Theta3
    c23 = (Px * math.cos(theta1_rad) + Py * math.sin(theta1_rad) - L1 - L2 * math.cos(theta2_2_rad) - L4 * math.cos(theta_rad)) / L3
    s23 = (Pz - d1 - L2 * math.sin(theta2_2_rad) - L4 * math.sin(theta_rad)) / L3
    theta3_2_rad = math.atan2(s23, c23) - theta2_2_rad
    Theta3_2 = math.degrees(theta3_2_rad)  # Chuyển sang độ
    Theta3_2 = round(Theta3_2, 0)

    if Theta3_2 < -180:
        Theta3_2 = Theta3_2 + 360
    elif Theta3_2 > 180:
        Theta3_2 = Theta3_2 - 360

    # Theta4
    theta4_2_rad = theta_rad - theta2_2_rad - theta3_2_rad
    Theta4_2 = math.degrees(theta4_2_rad)  # Chuyển sang độ
    Theta4_2 = round(Theta4_2, 0)

    Thetas = [[Theta1, Theta2_1, Theta3_1, Theta4_1],
             [Theta1, Theta2_2, Theta3_2, Theta4_2]]
    return Thetas