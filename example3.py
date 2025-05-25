# Orientation estimation using 9 DoF IMU

import ekf
import sympy as sp

dt = sp.Symbol("T")

qw, qx, qy, qz = sp.symbols("q_w q_x q_y q_z")
thetad = sp.symbols("theta_d")
wx, wy, wz = sp.symbols("w_x w_y w_z")

q = sp.Quaternion(qw, qx, qy, qz, norm=1)
w = sp.Matrix([wx, wy, wz])

x = sp.Matrix([q.to_Matrix(), thetad,])

u = sp.Matrix([w,])

f = sp.Matrix([(q + 0.5 * dt * q * sp.Quaternion(0, wx, wy, wz)).to_Matrix(), thetad,])

h_mag = q.to_rotation_matrix().transpose() * sp.Matrix(
    [0, sp.cos(thetad), sp.sin(thetad)]
)
h_acc = q.to_rotation_matrix().transpose() * sp.Matrix([0, 0, 1])

estimator = ekf.EKF(
    ekf.SystemModel(
        model=f,
        input=u,
        state=[(qw, 1, 1), (qx, 0, 1), (qy, 0, 1), (qz, 0, 1), (thetad, 0, 1),],
    ),
    [
        ekf.MeasurementModel(name="magnetometer", model=h_mag, covariance=100,),
        ekf.MeasurementModel(name="accelerometer", model=h_acc, covariance=10000,),
    ],
    [(dt, 0.001),],
)

estimator.generate_src("generated")
estimator.generate_docs("generated")
