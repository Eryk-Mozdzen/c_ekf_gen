import numpy as np
import sympy as sp
import sympy.codegen.ast
import os
import datetime
import sys

class SystemModel:
    def __init__(self, model, input, state):
        self.model = model
        self.input = input
        self.state = sp.Matrix([t[0] for t in state])
        self.state_elements = [sp.ccode(t[0]).replace('_', '') for t in state]
        self.covariance = [t[2] for t in state]
        self.initial_state = [t[1] for t in state]

    def u_dim(self):
        return self.input.shape[0] #TODO

    def x_dim(self):
        return self.state.shape[0]

class MeasurementModel:
    def __init__(self, model, covariance, name):
        self.model = model
        self.covariance = covariance
        self.name = name

    def dim(self):
        return self.model.shape[0]

class EKF:
    def __init__(self, system_model, measurement_models, parameters=[]):
        self.system = system_model
        self.measurements = measurement_models
        self.parameters = parameters

    def generate_src(self, path):
        path = os.path.normpath(os.path.dirname(sys._getframe(1).f_globals.get('__file__')) + '/' + path)
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

        assert len(self.system.initial_state) == x_dim

        with open(os.path.join(path, 'estimator.h'), 'w') as file:
            file.write(EKF.__header())
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

            if len(self.parameters)>0:
                file.write('\n')
                for param in self.parameters:
                    file.write(f'extern float {param[0].name};\n')

            file.write('\n')
            file.write('#define ESTIMATOR_PREDICT(u_data)')
            for i in range(padding + 1):
                file.write(' ')
            file.write(f' ekf_predict_{x_dim}_{u_dim}(&ekf, &system_model, u_data)\n')

            file.write('\n')
            for measurement in self.measurements:
                z_dim = measurement.dim()
                file.write('#define ESTIMATOR_CORRECT_' + measurement.name.upper() + '(z_data)')
                for i in range(padding - len(measurement.name)):
                    file.write(' ')
                file.write(f' ekf_correct_{x_dim}_{z_dim}(&ekf, &{measurement.name}_model, z_data)\n')

            file.write('\n')
            for i, element in enumerate(self.system.state_elements):
                file.write('#define ESTIMATOR_GET_' + element.upper() + '()')
                for j in range(padding - len(element) + 11):
                    file.write(' ')
                file.write('(ekf.x.pData[' + str(i) + '])\n')

            file.write('\n')
            for i, element in enumerate(self.system.state_elements):
                file.write('#define ESTIMATOR_SET_' + element.upper() + '(val)')
                for j in range(padding - len(element) + 8):
                    file.write(' ')
                file.write('do { ekf.x.pData[' + str(i) + '] = (val); } while(0)\n')

            file.write('\n')
            file.write(f'EKF_PREDICT_DEF({x_dim}, {u_dim})\n')
            for z_dim in z_dims:
                file.write(f'EKF_CORRECT_DEF({x_dim}, {z_dim})\n')

            file.write(
                '\n'
                '#endif\n'
            )

        with open(os.path.join(path, 'estimator.c'), 'w') as file:
            functions = {
                'Pow': [
                    (lambda base, exponent: exponent==2, lambda base, exponent: '(%s)*(%s)' % (base, base)),
                    (lambda base, exponent: exponent==-1, lambda base, exponent: '(1.F/(%s))' % (base)),
                    (lambda base, exponent: exponent!=2, lambda base, exponent: 'powf(%s, %s)' % (base, exponent))
                ],
            }

            aliases = {
                sympy.codegen.ast.real: sympy.codegen.ast.float32,
            }

            def estimator():
                initial = np.array(self.system.initial_state).reshape(-1, 1)
                covariance = np.eye(x_dim)

                file.write(EKF.__matrix(initial, 'x_data'))
                file.write(EKF.__matrix(covariance, 'P_data'))

                file.write(
                    'ekf_t ekf = {\n'
                    '\t.x.numRows = ' + str(x_dim) + ',\n'
                    '\t.x.numCols = 1,\n'
                    '\t.x.pData = x_data,\n'
                    '\t.P.numRows = ' + str(x_dim) + ',\n'
                    '\t.P.numCols = ' + str(x_dim) + ',\n'
                    '\t.P.pData = P_data,\n'
                    '};\n'
                    '\n'
                )

            def system_model(model, variance):
                assert len(variance) == x_dim

                subs = []
                subs.extend([(xx, sp.Symbol(f'x[{i}]')) for i, xx in enumerate(x)])
                subs.extend([(uu, sp.Symbol(f'u[{i}]')) for i, uu in enumerate(u)])

                cse_subs, cse_reduced = sp.cse([model], optimizations='basic')
                model_reduced, = cse_reduced

                file.write('static void system_f(const float *x, const float *u, float *x_next) {\n')
                for lhs, rhs in cse_subs:
                    file.write(f'\tconst float {lhs} = {sp.ccode(rhs.subs(subs), user_functions=functions, type_aliases=aliases)};\n')
                if len(cse_subs)>0:
                    file.write('\n')
                for i in range(x_dim):
                    file.write(f'\tx_next[{i}] = {sp.ccode(model_reduced[i].subs(subs), user_functions=functions, type_aliases=aliases)};\n')
                file.write('}\n')
                file.write('\n')

                cse_subs, cse_reduced = sp.cse([model.jacobian(x)], optimizations='basic')
                model_reduced, = cse_reduced

                file.write('static void system_df(const float *x, const float *u, float *x_next) {\n')
                for lhs, rhs in cse_subs:
                    file.write(f'\tconst float {lhs} = {sp.ccode(rhs.subs(subs), user_functions=functions, type_aliases=aliases)};\n')
                if len(cse_subs)>0:
                    file.write('\n')
                for i in range(x_dim):
                    for j in range(x_dim):
                        file.write(f'\tx_next[{i*x_dim + j}] = {sp.ccode(model_reduced[i, j].subs(subs), user_functions=functions, type_aliases=aliases)};\n')
                    if i!=(x_dim-1):
                        file.write('\n')
                file.write('}\n')
                file.write('\n')

                file.write(EKF.__matrix(np.diag(variance), 'system_Q_data'))

                file.write(
                    'ekf_system_model_t system_model = {\n'
                    '\t.Q.numRows = ' + str(x_dim) + ',\n'
                    '\t.Q.numCols = ' + str(x_dim) + ',\n'
                    '\t.Q.pData = system_Q_data,\n'
                    '\t.f = system_f,\n'
                    '\t.df = system_df,\n'
                    '};\n'
                    '\n'
                )

            def measurement_model(name, model, variance):
                x_dim = x.shape[0]
                z_dim = model.shape[0]

                subs = [(xx, sp.Symbol(f'x[{i}]')) for i, xx in enumerate(x)]

                cse_subs, cse_reduced = sp.cse([model], optimizations='basic')
                model_reduced, = cse_reduced

                file.write(f'static void {name}_h(const float *x, float *z) {{\n')
                for lhs, rhs in cse_subs:
                    file.write(f'\tconst float {lhs} = {sp.ccode(rhs.subs(subs), user_functions=functions, type_aliases=aliases)};\n')
                if len(cse_subs)>0:
                    file.write('\n')
                for i in range(z_dim):
                    file.write(f'\tz[{i}] = {sp.ccode(model_reduced[i].subs(subs), user_functions=functions, type_aliases=aliases)};\n')
                file.write('}\n')
                file.write('\n')

                cse_subs, cse_reduced = sp.cse([model.jacobian(x)], optimizations='basic')
                model_reduced, = cse_reduced

                file.write(f'static void {name}_dh(const float *x, float *z) {{\n')
                for lhs, rhs in cse_subs:
                    file.write(f'\tconst float {lhs} = {sp.ccode(rhs.subs(subs), user_functions=functions, type_aliases=aliases)};\n')
                if len(cse_subs)>0:
                    file.write('\n')
                for i in range(z_dim):
                    for j in range(x_dim):
                        file.write(f'\tz[{i*x_dim + j}] = {sp.ccode(model_reduced[i, j].subs(subs), user_functions=functions, type_aliases=aliases)};\n')
                    if i!=(z_dim-1):
                        file.write('\n')
                file.write('}\n')
                file.write('\n')

                file.write(EKF.__matrix(variance*np.eye(z_dim), name + '_R_data'))

                file.write(
                    'ekf_measurement_model_t ' + name + '_model = {\n'
                    '\t.R.numRows = ' + str(z_dim) + ',\n'
                    '\t.R.numCols = ' + str(z_dim) + ',\n'
                    '\t.R.pData = ' + name + '_R_data,\n'
                    '\t.h = ' + name + '_h,\n'
                    '\t.dh = ' + name + '_dh,\n'
                    '};\n'
                    '\n'
                )

            file.write(EKF.__header())
            file.write(
                '#include <math.h>\n'
                '\n'
                '#include "ekf.h"\n'
                '\n'
            )

            if len(self.parameters)>0:
                for param in self.parameters:
                    file.write(f'float {param[0].name} = {param[1]:f}f;\n')
                file.write('\n')

            estimator()
            system_model(self.system.model, self.system.covariance)
            for measurement in self.measurements:
                measurement_model(measurement.name, measurement.model, measurement.covariance)

            file.write(f'EKF_PREDICT({x_dim}, {u_dim})\n')
            for z_dim in z_dims:
                file.write(f'EKF_CORRECT({x_dim}, {z_dim})\n')

    def generate_docs(self, path, compile=True):
        path = os.path.normpath(os.path.dirname(sys._getframe(1).f_globals.get('__file__')) + '/' + path)
        os.makedirs(path, exist_ok=True)

        x = self.system.state

        with open(os.path.join(path, 'estimator.tex'), 'w') as file:
            file.write(EKF.__header(comment='%'))
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
                '\t\\[\\frac{\\partial}{\\partial x}f(x_{k-1}, u_k) = ' + sp.latex(self.system.model.jacobian(x)) + '\\]\n'
            )

            for measurement in self.measurements:
                file.write(f'\t\\[h_{{{measurement.name}}}(x_k) = {sp.latex(measurement.model)}\\]\n')
                file.write(f'\t\\[\\frac{{\\partial}}{{\\partial x}}h_{{{measurement.name}}}(x_k) = {sp.latex(measurement.model.jacobian(x))}\\]\n')

            file.write('\\end{document}\n')

        if compile:
            os.system(f'pdflatex -interaction=nonstopmode -output-directory={path} {os.path.join(path, "estimator.tex")} > /dev/null')
            os.system(f'rm {os.path.join(path, "estimator.aux")}')
            os.system(f'rm {os.path.join(path, "estimator.log")}')
            os.system(f'rm {os.path.join(path, "estimator.tex")}')

    def __header(comment='//'):
        text = \
        '{comment} auto-generated\n' \
        '{comment} {time} {date}\n' \
        '\n'

        return text.format(
            comment=comment,
            time=datetime.datetime.now().strftime('%H:%M:%S'),
            date=datetime.datetime.now().strftime('%d-%m-%Y'),
        )

    def __matrix(matrix, name):
        rows = matrix.shape[0]
        cols = matrix.shape[1]

        text = f'static float {name}[{rows*cols}] = {{\n'
        for i in range(rows):
            text +='\t'
            for j in range(cols):
                text +=f'{int(matrix[i][j])},'
                if j!=(cols-1):
                    text +=' '
            text +='\n'
        text +='};\n\n'

        return text
