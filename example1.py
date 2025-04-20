# UAV example

import sympy as sp
import scipy.constants
import ekf

dt = sp.Symbol('T')
g = sp.Symbol('g')

qw, qx, qy, qz = sp.symbols('q_w q_x q_y q_z')
Wx, Wy, Wz = sp.symbols('omega_x omega_y omega_z')
px, py, pz = sp.symbols('p_x p_y p_z')
vx, vy, vz = sp.symbols('v_x v_y v_z')
thetad = sp.symbols('theta_d')
p0 = sp.symbols('p_0')
wx, wy, wz = sp.symbols('w_x w_y w_z')
ax, ay, az = sp.symbols('a_x a_y a_z')

q = sp.Quaternion(qw, qx, qy, qz, norm=1)
W = sp.Matrix([Wx, Wy, Wz])
p = sp.Matrix([px, py, pz])
v = sp.Matrix([vx, vy, vz])
w = sp.Matrix([wx, wy, wz])
a = sp.Matrix([ax, ay, az])

x = sp.Matrix([
    q.to_Matrix(),
    W,
    p,
    v,
    thetad,
    p0,
])

u = sp.Matrix([
    w,
    a,
])

f = sp.Matrix([
    (q + 0.5*dt*q*sp.Quaternion(0, wx, wy, wz)).to_Matrix(),
    w,
    p + dt*v - 0.5*dt**2*(q.to_rotation_matrix()*a - sp.Matrix([0, 0, -g])),
    v - dt*(q.to_rotation_matrix()*a - sp.Matrix([0, 0, -g])),
    thetad,
    p0,
])

h_mag = q.to_rotation_matrix().transpose()*sp.Matrix([0, sp.cos(thetad), sp.sin(thetad)])
h_range = sp.Matrix([pz])
h_press = sp.Matrix([p0*sp.Pow(1 - pz/44330, 5.255)])
h_gps = sp.Matrix([px, py])
yaw = sp.atan2(2 * (qw*qz + qx*qy), 1 - 2 * (qy**2 + qz**2))
h_flow = sp.Matrix([[sp.cos(yaw), sp.sin(yaw)], [-sp.sin(yaw), sp.cos(yaw)]])*sp.Matrix([vx/pz, vy/pz]) + sp.Matrix([-Wy, Wx])

estimator = ekf.EKF(
    ekf.SystemModel(
        model=f,
        input=u,
        state=[
            (qw, 1, 1),
            (qx, 0, 1),
            (qy, 0, 1),
            (qz, 0, 1),
            (Wx, 0, 1),
            (Wy, 0, 1),
            (Wz, 0, 1),
            (px, 0, 1),
            (py, 0, 1),
            (pz, 0, 1),
            (vx, 0, 1),
            (vy, 0, 1),
            (vz, 0, 1),
            (thetad, 0, 1),
            (p0, 102400, 1),
        ],
    ),
    [
        ekf.MeasurementModel(
            name='magnetometer',
            model=h_mag,
            covariance=100,
        ),
        ekf.MeasurementModel(
            name='rangefinder',
            model=h_range,
            covariance=10,
        ),
        ekf.MeasurementModel(
            name='barometer',
            model=h_press,
            covariance=100,
        ),
        ekf.MeasurementModel(
            name='gps',
            model=h_gps,
            covariance=100000,
        ),
        ekf.MeasurementModel(
            name='flow',
            model=h_flow,
            covariance=100000,
        ),
    ],
    [
        (dt, 0.001),
        (g, scipy.constants.g),
    ],
)

estimator.generate_src('generated')
estimator.generate_docs('generated')
