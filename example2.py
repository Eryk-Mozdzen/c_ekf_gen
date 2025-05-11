# mobile platform example

import sympy as sp
import ekf

dt = sp.Symbol('T')
fh = sp.Symbol('f_h')
fl = sp.Symbol('f_l')
theta0 = sp.symbols('theta0')

px, py, theta = sp.symbols('px py theta')
vx, vy, vtheta = sp.symbols('vx vy vtheta')
wx, wy, wz = sp.symbols('w_x w_y w_z')
ax, ay, az = sp.symbols('a_x a_y a_z')
mx, my, mz, m0 = sp.symbols('m_x m_y m_z m0')

p = sp.Matrix([px, py])
v = sp.Matrix([vx, vy])
a = sp.Matrix([ax, ay])
m = sp.Matrix([mx, my, mz])

u = sp.Matrix([
    ax,
    ay,
    wz,
])

def rot2d(angle):
    return sp.Matrix([
        [sp.cos(angle), sp.sin(angle)],
        [-sp.sin(angle), sp.cos(angle)],
    ])

def rot3d(angle):
    return sp.Matrix([
        [sp.cos(angle), sp.sin(angle), 0],
        [-sp.sin(angle), sp.cos(angle), 0],
        [0, 0, 1],
    ])

f = sp.Matrix([
    p + dt*v - 0.5*dt**2*rot2d(theta)*a,
    theta + dt*wz,
    v - dt*rot2d(theta)*a,
    wz,
    m0,
])

h_mag = rot3d(theta + theta0)*sp.Matrix([0, sp.cos(m0), sp.sin(m0)])
h_flow = rot2d(theta)*sp.Matrix([vx/fh, (vy - vtheta*fl)/fh])

estimator = ekf.EKF(
    ekf.SystemModel(
        model=f,
        input=u,
        state=[
            (px, 0, 1),
            (py, 0, 1),
            (theta, 0, 1),
            (vx, 0, 1),
            (vy, 0, 1),
            (vtheta, 0, 1),
            (m0, 0, 1),
        ],
    ),
    [
        ekf.MeasurementModel(
            name='magnetometer',
            model=h_mag,
            covariance=100,
        ),
        ekf.MeasurementModel(
            name='flow',
            model=h_flow,
            covariance=1000,
        ),
    ],
    [
        (dt, 0.001),
        (fh, 0.065),
        (fl, 0.095),
        (theta0, 0),
    ],
)

estimator.generate_src('generated')
estimator.generate_docs('generated')
