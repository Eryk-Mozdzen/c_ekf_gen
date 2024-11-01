import numpy as np
import sympy as sp
import sympy.codegen.ast
import os
import datetime

class SystemModel:
    def __init__(self, model, state, input, covariance):
        self.model = model
        self.state = state
        self.input = input
        self.covariance = covariance

    def u_dim(self):
        return self.input.shape[0] #TODO

    def x_dim(self):
        return self.state.shape[0]

class MeasurementModel:
    def __init__(self, model, covariance, name, name_short=None):
        self.model = model
        self.covariance = covariance
        self.name = name

        if name_short:
            self.name_short = name_short
        elif len(name)<=5:
            self.name_short = name
        else:
            self.name_short = name[:min(len(name), 3)]

    def dim(self):
        return self.model.shape[0]

class EKF:
    def __init__(self, system_model, measurement_models, initial_state, parameters=[]):
        self.system = system_model
        self.measurements = measurement_models
        self.initial_state = initial_state
        self.parameters = parameters

    def generate_source(self, path):
        path = os.path.join(os.path.dirname(__file__), path)
        os.makedirs(path, exist_ok=True)

        padding = 0
        z_dims = []
        for measurement in self.measurements:
            z = measurement.dim()
            if not z in z_dims:
                z_dims.append(z)
            padding = max(padding, len(measurement.name))
        x_dim = self.system.x_dim()
        u_dim = self.system.u_dim()
        z_dims = np.sort(z_dims)

        x = self.system.state
        u = self.system.input

        with open(os.path.join(path, 'estimator.h'), 'w') as file:
            EKF.__header(file)

            file.write(
                '#ifndef ESTIMATOR_H\n'
                '#define ESTIMATOR_H\n'
                '\n'
                '#include "ekf.h"\n'
                '\n'
                'extern ekf_t ekf;\n'
                'extern ekf_system_model_t system_model;\n'
            )

            for measurement in self.measurements:
                file.write('extern ekf_measurement_model_t ' + measurement.name + '_model;\n')

            file.write(
                '\n'
                'EKF_PREDICT_DEF(' + str(self.system.x_dim()) + ', ' + str(self.system.u_dim()) + ')\n'
            )

            for z_dim in z_dims:
                file.write('EKF_CORRECT_DEF(' + str(x_dim) + ', ' + str(z_dim) + ')\n')

            file.write('\n')
            file.write('#define ESTIMATOR_PREDICT_SYSTEM(u_data)')
            for i in range(padding - len('system')):
                file.write(' ')
            file.write(' EKF_PREDICT_' + str(x_dim) + '_' + str(u_dim) + '(&ekf, &system_model, u_data)\n')

            for measurement in self.measurements:
                z_dim = measurement.dim()
                file.write('#define ESTIMATOR_CORRECT_' + measurement.name.upper() + '(z_data)')
                for i in range(padding - len(measurement.name)):
                    file.write(' ')
                file.write(' EKF_CORRECT_' + str(x_dim) + '_' + str(z_dim) + '(&ekf, &' + measurement.name + '_model, z_data)\n')

            file.write(
                '\n'
                '#endif\n'
            )

        with open(os.path.join(path, 'estimator.c'), 'w') as file:
            functions = {
                'Pow': [
                    (lambda base, exponent: exponent==2, lambda base, exponent: '(%s)*(%s)' % (base, base)),
                    (lambda base, exponent: exponent!=2, lambda base, exponent: 'powf(%s, %s)' % (base, exponent))
                ],
            }

            aliases = {
                sympy.codegen.ast.real: sympy.codegen.ast.float32,
            }

            def estimator(initial, variance):
                assert len(initial) == x_dim

                file.write('static float x_data[' + str(x_dim) + '] = {\n')
                file.write('\t')
                for i in range(x_dim):
                    file.write(str(initial[i]) + ',')
                    if i!=(x_dim-1):
                        file.write(' ')
                file.write('\n')
                file.write('};\n')
                file.write('\n')
                file.write('static float P_data[' + str(x_dim**2) + '] = {\n')
                for i in range(x_dim):
                    file.write('\t')
                    for j in range(x_dim):
                        if i==j:
                            file.write(str(variance) + ',')
                        else:
                            file.write('0,')
                        if j!=(x_dim-1):
                            file.write(' ')
                    file.write('\n')
                file.write('};\n')
                file.write('\n')
                file.write('ekf_t ekf = {\n')
                file.write('\t.x.numRows = ' + str(x_dim) + ',\n')
                file.write('\t.x.numCols = 1,\n')
                file.write('\t.x.pData = x_data,\n')
                file.write('\t.P.numRows = ' + str(x_dim) + ',\n')
                file.write('\t.P.numCols = ' + str(x_dim) + ',\n')
                file.write('\t.P.pData = P_data,\n')
                file.write('};\n')
                file.write('\n')

            def system_model(model, variance):
                assert len(variance) == x_dim

                f_used = list(model.free_symbols)
                df_used = list(model.jacobian(x).free_symbols)

                file.write('static void system_f(const float *x, const float *u, float *x_next) {\n')
                for i in range(u_dim):
                    if u[i] in f_used:
                        file.write('\tconst float ' + sympy.ccode(u[i]) + ' = u[' + str(i) + '];\n')
                if len(f_used)>0:
                    file.write('\n')
                for i in range(x_dim):
                    if x[i] in f_used:
                        file.write('\tconst float ' + sympy.ccode(x[i]) + ' = x[' + str(i) + '];\n')
                if len(f_used)>0:
                    file.write('\n')
                for i in range(x_dim):
                    file.write('\tx_next[' + str(i) + '] = ' + sympy.ccode(model[i], user_functions=functions, type_aliases=aliases) + ';\n')
                file.write('}\n')
                file.write('\n')
                file.write('static void system_df(const float *x, const float *u, float *x_next) {\n')
                for i in range(u_dim):
                    if u[i] in df_used:
                        file.write('\tconst float ' + sympy.ccode(u[i]) + ' = u[' + str(i) + '];\n')
                if len(f_used)>0:
                    file.write('\n')
                for i in range(x_dim):
                    if x[i] in df_used:
                        file.write('\tconst float ' + sympy.ccode(x[i]) + ' = x[' + str(i) + '];\n')
                if len(df_used)>0:
                    file.write('\n')
                for i in range(x_dim):
                    for j in range(x_dim):
                        file.write('\tx_next[' + str(i*x_dim + j) + '] = ' + sympy.ccode(model.jacobian(x)[i, j], user_functions=functions, type_aliases=aliases) + ';\n')
                    if i!=(x_dim-1):
                        file.write('\n')
                file.write('}\n')
                file.write('\n')
                file.write('static float system_Q_data[' + str(x_dim**2) + '] = {\n')
                for i in range(x_dim):
                    file.write('\t')
                    for j in range(x_dim):
                        if i==j:
                            file.write(str(variance[i]) + ',')
                        else:
                            file.write('0,')
                        if j!=(x_dim-1):
                            file.write(' ')
                    file.write('\n')
                file.write('};\n')
                file.write('\n')
                file.write('ekf_system_model_t system_model = {\n')
                file.write('\t.Q.numRows = ' + str(x_dim) + ',\n')
                file.write('\t.Q.numCols = ' + str(x_dim) + ',\n')
                file.write('\t.Q.pData = system_Q_data,\n')
                file.write('\t.f = system_f,\n')
                file.write('\t.df = system_df,\n')
                file.write('};\n')
                file.write('\n')

            def measurement_model(name, model, variance):
                x_dim = x.shape[0]
                z_dim = model.shape[0]
                h_used = list(model.free_symbols)
                dh_used = list(model.jacobian(x).free_symbols)

                file.write('static void ' + name + '_h(const float *x, float *z) {\n')
                for i in range(x_dim):
                    if x[i] in h_used:
                        file.write('\tconst float ' + sympy.ccode(x[i]) + ' = x[' + str(i) + '];\n')
                if len(h_used)>0:
                    file.write('\n')
                for i in range(z_dim):
                    file.write('\tz[' + str(i) + '] = ' + sympy.ccode(model[i], user_functions=functions, type_aliases=aliases) + ';\n')
                file.write('}\n')
                file.write('\n')
                file.write('static void ' + name + '_dh(const float *x, float *z) {\n')
                for i in range(x_dim):
                    if x[i] in dh_used:
                        file.write('\tconst float ' + sympy.ccode(x[i]) + ' = x[' + str(i) + '];\n')
                if len(dh_used)>0:
                    file.write('\n')
                for i in range(z_dim):
                    for j in range(x_dim):
                        file.write('\tz[' + str(i*x_dim + j) + '] = ' + sympy.ccode(model.jacobian(x)[i, j], user_functions=functions, type_aliases=aliases) + ';\n')
                    if i!=(z_dim-1):
                        file.write('\n')
                file.write('}\n')
                file.write('\n')
                file.write('static float ' + name + '_R_data[' + str(z_dim**2) + '] = {\n')
                for i in range(z_dim):
                    file.write('\t')
                    for j in range(z_dim):
                        if i==j:
                            file.write(str(variance) + ',')
                        else:
                            file.write('0,')
                        if j!=(z_dim-1):
                            file.write(' ')
                    file.write('\n')
                file.write('};\n')
                file.write('\n')
                file.write('ekf_measurement_model_t ' + name + '_model = {\n')
                file.write('\t.R.numRows = ' + str(z_dim) + ',\n')
                file.write('\t.R.numCols = ' + str(z_dim) + ',\n')
                file.write('\t.R.pData = ' + name + '_R_data,\n')
                file.write('\t.h = ' + name + '_h,\n')
                file.write('\t.dh = ' + name + '_dh,\n')
                file.write('};\n')
                file.write('\n')

            EKF.__header(file)
            file.write(
                '#include <math.h>\n'
                '\n'
                '#include "ekf.h"\n'
                '\n'
            )
            if len(self.parameters)>0:
                for param in self.parameters:
                    file.write('#define ' + param[0].name + ' ' + str(param[1]) + '\n')
                file.write('\n')
            estimator(self.initial_state, 1)
            system_model(self.system.model, self.system.covariance)
            for measurement in self.measurements:
                measurement_model(measurement.name, measurement.model, measurement.covariance)
            file.write('EKF_PREDICT(' + str(x_dim) + ', ' + str(u_dim) + ')\n')
            for z_dim in z_dims:
                file.write('EKF_CORRECT(' + str(x_dim) + ', ' + str(z_dim) + ')\n')

    def generate_docs(self, path, compile=True):
        path = os.path.join(os.path.dirname(__file__), path)
        os.makedirs(path, exist_ok=True)

        x = self.system.state

        with open(os.path.join(path, 'estimator.tex'), 'w') as file:
            EKF.__header(file, comment='%')
            file.write(
                '\\documentclass{article}\n'
                '\\usepackage{geometry}\n'
                '\\usepackage{amsmath}\n'
                '\n'
                '\\geometry{\n'
                '    paperwidth=50cm,\n'
                '    paperheight=50cm,\n'
                '    margin=10mm\n'
                '}\n'
                '\n'
                '\\begin{document}\n'
                '\t\\[x_k = ' + sp.latex(x) + ' = f(x_{k-1}, u_k) = ' + sp.latex(self.system.model) + '\\]\n'
                '\t\\[\\frac{\partial}{\partial x}f(x_{k-1}, u_k) = ' + sp.latex(self.system.model.jacobian(x)) + '\\]\n'
            )

            for measurement in self.measurements:
                file.write(
                    '\t\\[h_{' + measurement.name_short + '}(x_k) = ' + sp.latex(measurement.model) + '\\]\n'
                    '\t\\[\\frac{\partial}{\partial x}h_{' + measurement.name_short + '}(x_k) = ' + sp.latex(measurement.model.jacobian(x)) + '\\]\n'
                )

            file.write(
                '\\end{document}\n'
            )

        if compile:
            os.system('pdflatex -interaction=nonstopmode -output-directory=' + path + ' ' + os.path.join(path, 'estimator.tex > /dev/null'))
            os.system('rm ' + os.path.join(path, 'estimator.aux'))
            os.system('rm ' + os.path.join(path, 'estimator.log'))
            os.system('rm ' + os.path.join(path, 'estimator.tex'))

    def __header(file, comment='//'):
        file.write(comment + ' auto-generated\n')
        file.write(comment + ' ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
        file.write('\n')
