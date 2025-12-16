import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sympy as sp
import math
from sympy import *
import re

def polinomio_taylor(func_str, a, n, variable='x'):
    try:
        x = sp.symbols(variable)

        func_str = normalizar_ecuacion_str(func_str)
        f = sp.sympify(func_str, locals=FUNCIONES_DISPONIBLES)

        poly = 0
        coeficientes = []

        for k in range(n + 1):
            deriv = sp.diff(f, x, k)
            coef = deriv.subs(x, a) / sp.factorial(k)
            coeficientes.append(sp.simplify(coef))
            poly += coef * (x - a)**k

        poly = sp.expand(poly)
        func = sp.lambdify(x, poly, 'numpy')

        return poly, func, coeficientes, None

    except Exception as e:
        return None, None, None, f"Error al calcular Taylor: {e}"


def normalizar_ecuacion_str(s: str) -> str:
    """Normaliza la cadena de entrada para insertar multiplicaciones implícitas.
    Convierte '^' en '**' y añade '*' donde falte entre número/')' y variable/'('.
    """
    if s is None:
        return s
    s = s.strip()
    # convertir potencias
    s = s.replace('^', '**')
    # insertar '*' entre dígito o ')' y letra o '('
    s = re.sub(r'(?<=\d|\))\s*(?=[A-Za-z(])', '*', s)
    # insertar '*' entre letra y dígito (p.ej. x2 -> x*2)
    s = re.sub(r'(?<=[A-Za-z])\s*(?=\d)', '*', s)
    return s

# Diccionario de funciones matemáticas disponibles
FUNCIONES_DISPONIBLES = {
    # Trigonométricas
    'sin': sin, 'cos': cos, 'tan': tan, 'cot': cot, 'sec': sec, 'csc': csc,
    'asin': asin, 'acos': acos, 'atan': atan, 'acot': acot, 'asec': asec, 'acsc': acsc,
    'sinh': sinh, 'cosh': cosh, 'tanh': tanh, 'asinh': asinh, 'acosh': acosh, 'atanh': atanh,
    # Exponenciales y logarítmicas
    'exp': exp, 'log': log, 'ln': log, 'log10': log, 'sqrt': sqrt, 'cbrt': cbrt,
    # Otras
    'abs': Abs, 'Abs': Abs, 'ceiling': ceiling, 'floor': floor,
    'pi': pi, 'E': E, 'e': E, 'I': I,
    # Trigonométricas adicionales
    'factorial': factorial, 'gamma': gamma, 'erf': erf,
}

# Configurar estilos
def configurar_estilos():
    style = ttk.Style()
    style.theme_use('clam')
    
    # Colores modernos
    color_primario = "#2E86AB"
    color_secundario = "#A23B72"
    color_fondo = "#F5F7FA"
    color_texto = "#1A1A1A"
    color_boton = "#06A77D"
    color_boton_hover = "#048A5E"
    
    # Estilo para los notebooks
    style.configure('TNotebook', background=color_fondo, borderwidth=0)
    style.configure('TNotebook.Tab', padding=[20, 15], font=('Segoe UI', 10, 'bold'))
    style.map('TNotebook.Tab',
        background=[('selected', color_primario)],
        foreground=[('selected', 'white'), ('', color_texto)])
    
    # Estilo para labels
    style.configure('TLabel', background=color_fondo, font=('Segoe UI', 10), foreground=color_texto)
    style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'), foreground=color_primario, background=color_fondo)
    style.configure('Subtitle.TLabel', font=('Segoe UI', 11), foreground=color_secundario, background=color_fondo)
    
    # Estilo para entries
    style.configure('TEntry', fieldbackground='white', font=('Segoe UI', 10))
    
    # Estilo para botones
    style.configure('TButton', font=('Segoe UI', 11, 'bold'), padding=[10, 8])
    style.map('TButton',
        background=[('active', color_boton_hover), ('', color_boton)],
        foreground=[('', 'white')])
    
    # Estilo para frames
    style.configure('TFrame', background=color_fondo)

def parsear_funcion(func_str, variable='x'):
    """Parsea una cadena de función y retorna una función evaluable"""
    try:
        # Limpiar espacios
        func_str = func_str.strip()
        
        # Convertir ^ a ** para potencias
        func_str = func_str.replace('^', '**')
        
        # Crear símbolo
        x = sp.symbols(variable)
        
        # Convertir a expresión simbólica
        expr = sp.sympify(func_str, locals=FUNCIONES_DISPONIBLES)
        
        # Crear función lambda con manejo robusto
        f = sp.lambdify(x, expr, modules=[FUNCIONES_DISPONIBLES, 'numpy', 'sympy'])
        
        return f, expr, None
    except Exception as e:
        return None, None, f"Error al parsear: {str(e)}"

def parsear_matriz_ecuaciones(matriz_str, var_dict=None):
    """Parsea una cadena que representa una matriz de ecuaciones."""
    try:
        filas_str = [s.strip() for s in matriz_str.split(';') if s.strip()]
        matriz = []
        expresiones = []

        for fila_str in filas_str:
            elementos = [e.strip() for e in fila_str.split(',') if e.strip()]
            fila_vals = []
            fila_expr = []
            for elem in elementos:
                elem = elem.replace('^', '**')
                # intentar convertir a número
                try:
                    valor = float(elem)
                    fila_vals.append(valor)
                    fila_expr.append(None)
                except Exception:
                    # intentar parsear como expresión simbólica
                    expr = sp.sympify(elem, locals=FUNCIONES_DISPONIBLES)
                    fila_expr.append(expr)
                    if var_dict is not None:
                        try:
                            valor = float(expr.subs(var_dict))
                        except Exception:
                            valor = np.nan
                        fila_vals.append(valor)
                    else:
                        fila_vals.append(np.nan)
            matriz.append(fila_vals)
            expresiones.append(fila_expr)

        return np.array(matriz, dtype=float), expresiones, None
    except Exception as e:
        return None, None, f"Error al parsear matriz: {e}"

# Métodos numéricos
def biseccion(f, a, b, tol=1e-6, max_iter=100):
    try:
        fa = f(a)
        fb = f(b)
        if fa * fb > 0:
            return None, "f(a) y f(b) tienen el mismo signo"
        for k in range(max_iter):
            m = (a + b) / 2
            fm = f(m)
            if abs(fm) < tol:
                return m, f"Convergió en {k} iteraciones"
            if fa * fm < 0:
                b = m
                fb = fm
            else:
                a = m
                fa = fm
        return (a + b) / 2, f"Máximo de iteraciones alcanzado ({max_iter})"
    except Exception as e:
        return None, f"Error numérico: {str(e)}"

def newton_1var(f, df, x0, tol=1e-6, max_iter=50):
    try:
        x = x0
        for it in range(max_iter):
            fx = f(x)
            dfx = df(x)
            if abs(dfx) < 1e-12:
                return None, f"Derivada muy pequeña en iteración {it}"
            dx = fx / dfx
            x = x - dx
            if abs(dx) < tol:
                return x, f"Convergió en {it} iteraciones"
        return x, f"Máximo de iteraciones alcanzado ({max_iter})"
    except Exception as e:
        return None, f"Error numérico: {str(e)}"

def lagrange_interp(x, y, xval=None):
    """
    Interpolación de Lagrange.
    Si xval es None, retorna la función polinómica (como función lambda y coeficientes).
    Si xval es un valor, retorna el valor interpolado en ese punto.
    """
    n = len(x)
    
    # Convertir a símbolos para obtener la función polinómica
    try:
        import sympy as sp
        X = sp.Symbol('x')
        poly = 0
        
        for i in range(n):
            L = 1
            for j in range(n):
                if i != j:
                    L *= (X - x[j]) / (x[i] - x[j])
            poly += L * y[i]
        
        poly = sp.expand(poly)
        
        if xval is None:
            # Retornar la función simbólica y los coeficientes
            coeffs = sp.Poly(poly, X).all_coeffs()
            coeffs_float = [float(c) for c in coeffs]
            func = sp.lambdify(X, poly, 'numpy')
            return func, poly, coeffs_float
        else:
            # Retornar solo el valor evaluado
            func = sp.lambdify(X, poly, 'numpy')
            return float(poly.subs(X, xval))
    except:
        if xval is None:
            # Retornar función sin symbolic
            def func(val):
                total = 0
                for i in range(n):
                    L = 1
                    for j in range(n):
                        if i != j:
                            L *= (val - x[j]) / (x[i] - x[j])
                    total += L * y[i]
                return total
            return func, None, None
        else:
            total = 0
            for i in range(n):
                L = 1
                for j in range(n):
                    if i != j:
                        L *= (xval - x[j]) / (x[i] - x[j])
                total += L * y[i]
            return total

def diferencias_divididas(x, y):
    n = len(x)
    tabla = np.zeros((n, n))
    tabla[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            tabla[i][j] = (tabla[i+1][j-1] - tabla[i][j-1]) / (x[i+j] - x[i])
    return tabla

def jacobi(A, b, x0, tol=1e-6, max_iter=100):
    n = len(A)
    x = x0.copy()
    for it in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x

def gauss_seidel(A, b, x0, tol=1e-6, max_iter=100):
    n = len(A)
    x = x0.copy()
    for it in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i+1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x

def minimos_cuadrados(x, y):
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    A = np.vstack([x, np.ones(n)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, b

def diferencias_finitas(f, a, b, n, tipo='central'):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    if tipo == 'progresiva':
        x_df = x[:-1]
        df = (y[1:] - y[:-1]) / h

    elif tipo == 'regresiva':
        x_df = x[1:]
        df = (y[1:] - y[:-1]) / h

    elif tipo == 'central':
        x_df = x[1:-1]
        df = (y[2:] - y[:-2]) / (2 * h)

    else:
        raise ValueError("Tipo inválido")

    return x, y, x_df, df, h

# Resolvedores de ecuaciones polinomiales

def resolver_cuadratica(a, b, c):
    """
    Resuelve ecuación cuadrática: ax² + bx + c = 0
    Retorna (raíces, discriminante, mensaje)
    """
    try:
        if a == 0:
            if b == 0:
                return None, None, "No es una ecuación cuadrática válida (a=0, b=0)"
            raiz = -c / b
            return [raiz], 0, "Ecuación lineal"
        
        discriminante = b**2 - 4*a*c
        
        if discriminante > 0:
            raiz1 = (-b + np.sqrt(discriminante)) / (2*a)
            raiz2 = (-b - np.sqrt(discriminante)) / (2*a)
            return [raiz1, raiz2], discriminante, "Dos raíces reales distintas"
        elif discriminante == 0:
            raiz = -b / (2*a)
            return [raiz], discriminante, "Raíz doble (una raíz real)"
        else:
            # Raíces complejas
            parte_real = -b / (2*a)
            parte_imag = np.sqrt(-discriminante) / (2*a)
            raiz1 = complex(parte_real, parte_imag)
            raiz2 = complex(parte_real, -parte_imag)
            return [raiz1, raiz2], discriminante, "Dos raíces complejas conjugadas"
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def resolver_cubica(a, b, c, d):
    """
    Resuelve ecuación cúbica: ax³ + bx² + cx + d = 0
    Retorna (raíces, mensaje)
    """
    try:
        if a == 0:
            return resolver_cuadratica(b, c, d)[0], "Reducida a ecuación cuadrática"
        
        # Usar sympy para resolver la cúbica
        x = sp.symbols('x')
        ecuacion = a*x*3 + b*x*2 + c*x + d
        raices = sp.solve(ecuacion, x)
        
        # Convertir a números complejos o reales
        raices_numericas = []
        for raiz in raices:
            try:
                valor = complex(raiz.evalf())
                # Si la parte imaginaria es muy pequeña, mostrar como real
                if abs(valor.imag) < 1e-10:
                    raices_numericas.append(valor.real)
                else:
                    raices_numericas.append(valor)
            except:
                raices_numericas.append(float(raiz))
        
        return raices_numericas, "Cúbica resuelta"
    except Exception as e:
        return None, f"Error: {str(e)}"

def resolver_sistema_ecuaciones_2(eq1_str, eq2_str, var_dict_init=None):
    """
    Resuelve sistema de 2 ecuaciones no lineales
    eq1_str, eq2_str: Ecuaciones en términos de 'x' e 'y'
    Retorna (soluciones, mensaje)
    """
    try:
        x, y = sp.symbols('x y')
        eq1 = sp.sympify(eq1_str.replace('^', '**'), locals=FUNCIONES_DISPONIBLES)
        eq2 = sp.sympify(eq2_str.replace('^', '**'), locals=FUNCIONES_DISPONIBLES)
        
        # Resolver el sistema
        soluciones = sp.solve([eq1, eq2], [x, y])
        
        if not soluciones:
            return None, "No se encontraron soluciones reales"
        
        return soluciones, "Sistema resuelto exitosamente"
    except Exception as e:
        return None, f"Error: {str(e)}"

def resolver_sistema_ecuaciones_3(eq1_str, eq2_str, eq3_str):
    """
    Resuelve sistema de 3 ecuaciones no lineales
    Retorna (soluciones, mensaje)
    """
    try:
        x, y, z = sp.symbols('x y z')
        eq1 = sp.sympify(eq1_str.replace('^', '**'), locals=FUNCIONES_DISPONIBLES)
        eq2 = sp.sympify(eq2_str.replace('^', '**'), locals=FUNCIONES_DISPONIBLES)
        eq3 = sp.sympify(eq3_str.replace('^', '**'), locals=FUNCIONES_DISPONIBLES)
        
        # Resolver el sistema
        soluciones = sp.solve([eq1, eq2, eq3], [x, y, z])
        
        if not soluciones:
            return None, "No se encontraron soluciones reales"
        
        return soluciones, "Sistema resuelto exitosamente"
    except Exception as e:
        return None, f"Error: {str(e)}"

def resolver_sistema_newton_2(eq1_str, eq2_str, x0, y0, tol=1e-6, max_iter=50):
    """
    Resuelve sistema de 2 ecuaciones no lineales usando Newton multivariable
    Retorna (solución, iteraciones, mensaje)
    """
    try:
        x, y = sp.symbols('x y')
        eq1 = sp.sympify(eq1_str.replace('^', '**'), locals=FUNCIONES_DISPONIBLES)
        eq2 = sp.sympify(eq2_str.replace('^', '**'), locals=FUNCIONES_DISPONIBLES)
        
        # Calcular jacobiano
        J = sp.Matrix([
            [sp.diff(eq1, x), sp.diff(eq1, y)],
            [sp.diff(eq2, x), sp.diff(eq2, y)]
        ])
        
        # Crear funciones evaluables
        f1 = sp.lambdify((x, y), eq1, 'numpy')
        f2 = sp.lambdify((x, y), eq2, 'numpy')
        J_func = sp.lambdify((x, y), J, 'numpy')
        
        # Iteración de Newton
        xk = np.array([x0, y0], dtype=float)
        
        for it in range(max_iter):
            # Evaluar funciones
            Fk = np.array([f1(xk[0], xk[1]), f2(xk[0], xk[1])], dtype=float)
            
            # Evaluar jacobiano
            Jk = J_func(xk[0], xk[1]).astype(float)
            
            # Resolver sistema lineal
            try:
                delta = np.linalg.solve(Jk, -Fk)
            except np.linalg.LinAlgError:
                return None, it, "Matriz jacobiana singular"
            
            # Actualizar
            xk = xk + delta
            
            # Verificar convergencia
            if np.linalg.norm(delta) < tol:
                return xk, it, f"Convergió en {it} iteraciones"
        
        return xk, max_iter, f"Máximo de iteraciones alcanzado ({max_iter})"
    except Exception as e:
        return None, 0, f"Error: {str(e)}"

def resolver_sistema_newton_3(eq1_str, eq2_str, eq3_str, x0, y0, z0, tol=1e-6, max_iter=50):
    """
    Resuelve sistema de 3 ecuaciones no lineales usando Newton multivariable
    Retorna (solución, iteraciones, mensaje)
    """
    try:
        x, y, z = sp.symbols('x y z')
        eq1 = sp.sympify(eq1_str.replace('^', '**'), locals=FUNCIONES_DISPONIBLES)
        eq2 = sp.sympify(eq2_str.replace('^', '**'), locals=FUNCIONES_DISPONIBLES)
        eq3 = sp.sympify(eq3_str.replace('^', '**'), locals=FUNCIONES_DISPONIBLES)
        
        # Calcular jacobiano
        J = sp.Matrix([
            [sp.diff(eq1, x), sp.diff(eq1, y), sp.diff(eq1, z)],
            [sp.diff(eq2, x), sp.diff(eq2, y), sp.diff(eq2, z)],
            [sp.diff(eq3, x), sp.diff(eq3, y), sp.diff(eq3, z)]
        ])
        
        # Crear funciones evaluables
        f1 = sp.lambdify((x, y, z), eq1, 'numpy')
        f2 = sp.lambdify((x, y, z), eq2, 'numpy')
        f3 = sp.lambdify((x, y, z), eq3, 'numpy')
        J_func = sp.lambdify((x, y, z), J, 'numpy')
        
        # Iteración de Newton
        xk = np.array([x0, y0, z0], dtype=float)
        
        for it in range(max_iter):
            # Evaluar funciones
            Fk = np.array([f1(xk[0], xk[1], xk[2]), 
                          f2(xk[0], xk[1], xk[2]), 
                          f3(xk[0], xk[1], xk[2])], dtype=float)
            
            # Evaluar jacobiano
            Jk = J_func(xk[0], xk[1], xk[2]).astype(float)
            
            # Resolver sistema lineal
            try:
                delta = np.linalg.solve(Jk, -Fk)
            except np.linalg.LinAlgError:
                return None, it, "Matriz jacobiana singular"
            
            # Actualizar
            xk = xk + delta
            
            # Verificar convergencia
            if np.linalg.norm(delta) < tol:
                return xk, it, f"Convergió en {it} iteraciones"
        
        return xk, max_iter, f"Máximo de iteraciones alcanzado ({max_iter})"
    except Exception as e:
        return None, 0, f"Error: {str(e)}"

def resolver_sistema_newton_4(eq1_str, eq2_str, eq3_str, eq4_str, x0, y0, z0, w0, tol=1e-6, max_iter=50):
    """
    Resuelve sistema de 4 ecuaciones no lineales usando Newton multivariable
    Retorna (solución, iteraciones, mensaje)
    """
    try:
        x, y, z, w = sp.symbols('x y z w')
        eq1 = sp.sympify(eq1_str.replace('^', '**'), locals=FUNCIONES_DISPONIBLES)
        eq2 = sp.sympify(eq2_str.replace('^', '**'), locals=FUNCIONES_DISPONIBLES)
        eq3 = sp.sympify(eq3_str.replace('^', '**'), locals=FUNCIONES_DISPONIBLES)
        eq4 = sp.sympify(eq4_str.replace('^', '**'), locals=FUNCIONES_DISPONIBLES)
        
        # Calcular jacobiano
        J = sp.Matrix([
            [sp.diff(eq1, x), sp.diff(eq1, y), sp.diff(eq1, z), sp.diff(eq1, w)],
            [sp.diff(eq2, x), sp.diff(eq2, y), sp.diff(eq2, z), sp.diff(eq2, w)],
            [sp.diff(eq3, x), sp.diff(eq3, y), sp.diff(eq3, z), sp.diff(eq3, w)],
            [sp.diff(eq4, x), sp.diff(eq4, y), sp.diff(eq4, z), sp.diff(eq4, w)]
        ])
        
        # Crear funciones evaluables
        f1 = sp.lambdify((x, y, z, w), eq1, 'numpy')
        f2 = sp.lambdify((x, y, z, w), eq2, 'numpy')
        f3 = sp.lambdify((x, y, z, w), eq3, 'numpy')
        f4 = sp.lambdify((x, y, z, w), eq4, 'numpy')
        J_func = sp.lambdify((x, y, z, w), J, 'numpy')
        
        # Iteración de Newton
        xk = np.array([x0, y0, z0, w0], dtype=float)
        
        for it in range(max_iter):
            # Evaluar funciones
            Fk = np.array([f1(xk[0], xk[1], xk[2], xk[3]), 
                          f2(xk[0], xk[1], xk[2], xk[3]), 
                          f3(xk[0], xk[1], xk[2], xk[3]),
                          f4(xk[0], xk[1], xk[2], xk[3])], dtype=float)
            
            # Evaluar jacobiano
            Jk = J_func(xk[0], xk[1], xk[2], xk[3]).astype(float)
            
            # Resolver sistema lineal
            try:
                delta = np.linalg.solve(Jk, -Fk)
            except np.linalg.LinAlgError:
                return None, it, "Matriz jacobiana singular"
            
            # Actualizar
            xk = xk + delta
            
            # Verificar convergencia
            if np.linalg.norm(delta) < tol:
                return xk, it, f"Convergió en {it} iteraciones"
        
        return xk, max_iter, f"Máximo de iteraciones alcanzado ({max_iter})"
    except Exception as e:
        return None, 0, f"Error: {str(e)}"

# =====================
# Interfaz gráfica
# =====================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🔢 Simulación y Computación - Métodos Numéricos")
        self.geometry("1200x850")
        self.resizable(True, True)
        
        # Configurar estilos
        configurar_estilos()
        
        # Header
        header = tk.Frame(self, bg="#2E86AB", height=80)
        header.pack(fill='x', side='top')
        header.pack_propagate(False)
        
        title_label = tk.Label(header, text="🔢 Métodos Numéricos Aplicados", 
                              font=('Segoe UI', 22, 'bold'), fg='white', bg="#2E86AB")
        title_label.pack(pady=10)
        
        subtitle_label = tk.Label(header, text="Simulación y Computación Científica", 
                                 font=('Segoe UI', 11), fg='#A8D5E2', bg="#2E86AB")
        subtitle_label.pack()
        
        # Notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Diccionario para guardar referencias a los widgets de cada pestaña
        self.entries_por_pestana = {}
        self.resultados_por_pestana = {}
        
        self.crear_pestanas()
        
        # Bindar evento de cambio de pestaña
        self.notebook.bind("<<NotebookTabChanged>>", self.limpiar_pestana_anterior)

    def pestana_ecuaciones_polinomiales(self):
        # pestaña de polinomiales eliminada
        return

    def pestana_sistemas_ecuaciones(self):
        # pestaña de sistemas no lineales eliminada
        return

    def crear_pestanas(self):
        # pestañas eliminadas: polinomiales y sistemas no lineales
        self.pestana_taylor()
        self.pestana_biseccion()
        self.pestana_newton()
        self.pestana_lagrange()
        self.pestana_diferencias_divididas()
        self.pestana_jacobi()
        self.pestana_gauss_seidel()
        self.pestana_minimos_cuadrados()
        self.pestana_diferencias_finitas()
    def limpiar_pestana_anterior(self, event):
        """Limpia los campos de la pestaña anterior cuando se cambia"""
        current_index = self.notebook.index(self.notebook.select())
        
        # Limpiar cada pestaña cuando no está activa
        for pestana_nombre, entries in self.entries_por_pestana.items():
            for entry_widget in entries:
                entry_widget.delete(0, tk.END)
        
        for pestana_nombre, result_widget in self.resultados_por_pestana.items():
            result_widget.delete('1.0', tk.END)

    def pestana_biseccion(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📊 Bisección")
        
        # Canvas scrollable
        canvas = tk.Canvas(frame, bg="#F5F7FA", highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollable_frame.columnconfigure(0, weight=1)
        
        # Título
        title = ttk.Label(scrollable_frame, text="Método de Bisección", style='Title.TLabel')
        title.pack(pady=15)
        
        # Frame para inputs con mejor diseño
        input_frame = ttk.LabelFrame(scrollable_frame, text="Parámetros de entrada", padding=15)
        input_frame.pack(fill='x', padx=20, pady=10)
        input_frame.columnconfigure(1, weight=1)
        
        ttk.Label(input_frame, text="Función f(x):", style='Subtitle.TLabel').grid(row=0, column=0, sticky='w', pady=8)
        f_entry = ttk.Entry(input_frame, width=50)
        f_entry.grid(row=0, column=1, sticky='ew', padx=10)
        
        ttk.Label(input_frame, text="Límite a:", style='Subtitle.TLabel').grid(row=1, column=0, sticky='w', pady=8)
        a_entry = ttk.Entry(input_frame, width=50)
        a_entry.grid(row=1, column=1, sticky='ew', padx=10)
        
        ttk.Label(input_frame, text="Límite b:", style='Subtitle.TLabel').grid(row=2, column=0, sticky='w', pady=8)
        b_entry = ttk.Entry(input_frame, width=50)
        b_entry.grid(row=2, column=1, sticky='ew', padx=10)
        
        ttk.Label(input_frame, text="Tolerancia:", style='Subtitle.TLabel').grid(row=3, column=0, sticky='w', pady=8)
        tol_entry = ttk.Entry(input_frame, width=50)
        tol_entry.grid(row=3, column=1, sticky='ew', padx=10)
        
        ttk.Label(input_frame, text="Iteraciones máx:", style='Subtitle.TLabel').grid(row=4, column=0, sticky='w', pady=8)
        it_entry = ttk.Entry(input_frame, width=50)
        it_entry.grid(row=4, column=1, sticky='ew', padx=10)
        
        # Guardar referencias
        self.entries_por_pestana['biseccion'] = [f_entry, a_entry, b_entry, tol_entry, it_entry]
        
        # Frame para resultados
        result_frame = ttk.LabelFrame(scrollable_frame, text="Resultado", padding=15)
        result_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        resultado = tk.Text(result_frame, height=5, width=70, font=('Consolas', 10), bg='#F0F8FF', relief='flat')
        resultado.pack(fill='both', expand=True)
        
        # Guardar referencia al texto de resultado
        self.resultados_por_pestana['biseccion'] = resultado
        
        # Frame para botones
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(pady=20)
        
        def calcular():
            try:
                func_str = f_entry.get()
                a = float(a_entry.get())
                b = float(b_entry.get())
                tol = float(tol_entry.get())
                it = int(it_entry.get())
                
                # Parsear función
                f, expr, error = parsear_funcion(func_str)
                if error:
                    resultado.delete('1.0', tk.END)
                    resultado.insert(tk.END, f"❌ {error}\n\nFunción ingresada: {func_str}")
                    return
                
                res, mensaje = biseccion(f, a, b, tol, it)
                resultado.delete('1.0', tk.END)
                if res is None:
                    resultado.insert(tk.END, f"❌ {mensaje}\n\nFunción: {expr}")
                else:
                    resultado.insert(tk.END, f"✅ Raíz aproximada: {res:.15f}\n📊 {mensaje}\n\nFunción: {expr}")
            except ValueError as e:
                resultado.delete('1.0', tk.END)
                resultado.insert(tk.END, f"❌ Error en entrada numérica: {e}")
            except Exception as e:
                resultado.delete('1.0', tk.END)
                resultado.insert(tk.END, f"❌ Error inesperado: {e}")
        
        def limpiar():
            f_entry.delete(0, tk.END)
            a_entry.delete(0, tk.END)
            b_entry.delete(0, tk.END)
            tol_entry.delete(0, tk.END)
            it_entry.delete(0, tk.END)
            resultado.delete('1.0', tk.END)
        
        ttk.Button(button_frame, text="Calcular", command=calcular).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Limpiar", command=limpiar).pack(side='left', padx=5)
        
        # Empacar canvas y scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def pestana_newton(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📈 Newton-Raphson")
        
        # Canvas scrollable
        canvas = tk.Canvas(frame, bg="#F5F7FA", highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollable_frame.columnconfigure(0, weight=1)
        
        title = ttk.Label(scrollable_frame, text="Método de Newton-Raphson", style='Title.TLabel')
        title.pack(pady=15)
        
        input_frame = ttk.LabelFrame(scrollable_frame, text="Parámetros", padding=15)
        input_frame.pack(fill='x', padx=20, pady=10)
        input_frame.columnconfigure(1, weight=1)
        
        ttk.Label(input_frame, text="Función f(x):", style='Subtitle.TLabel').grid(row=0, column=0, sticky='w', pady=8)
        f_entry = ttk.Entry(input_frame, width=50)
        f_entry.grid(row=0, column=1, sticky='ew', padx=10)
        
        ttk.Label(input_frame, text="Derivada f'(x):", style='Subtitle.TLabel').grid(row=1, column=0, sticky='w', pady=8)
        df_entry = ttk.Entry(input_frame, width=50)
        df_entry.grid(row=1, column=1, sticky='ew', padx=10)
        
        ttk.Label(input_frame, text="x0 inicial:", style='Subtitle.TLabel').grid(row=2, column=0, sticky='w', pady=8)
        x0_entry = ttk.Entry(input_frame, width=50)
        x0_entry.grid(row=2, column=1, sticky='ew', padx=10)
        
        ttk.Label(input_frame, text="Tolerancia:", style='Subtitle.TLabel').grid(row=3, column=0, sticky='w', pady=8)
        tol_entry = ttk.Entry(input_frame, width=50)
        tol_entry.grid(row=3, column=1, sticky='ew', padx=10)
        
        ttk.Label(input_frame, text="Iteraciones máx:", style='Subtitle.TLabel').grid(row=4, column=0, sticky='w', pady=8)
        it_entry = ttk.Entry(input_frame, width=50)
        it_entry.grid(row=4, column=1, sticky='ew', padx=10)
        
        # Guardar referencias
        self.entries_por_pestana['newton'] = [f_entry, df_entry, x0_entry, tol_entry, it_entry]
        
        result_frame = ttk.LabelFrame(scrollable_frame, text="Resultado", padding=15)
        result_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        resultado = tk.Text(result_frame, height=5, width=70, font=('Consolas', 10), bg='#F0F8FF', relief='flat')
        resultado.pack(fill='both', expand=True)
        
        # Guardar referencia
        self.resultados_por_pestana['newton'] = resultado
        
        # Frame para botones
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(pady=20)
        
        def calcular_derivada():
            """Calcula automáticamente la derivada de f(x)"""
            try:
                func_str = f_entry.get().strip()
                if not func_str:
                    messagebox.showwarning("Advertencia", "Por favor ingresa primero la función f(x)")
                    return
                
                x = sp.symbols('x')
                expr = sp.sympify(func_str, locals=FUNCIONES_DISPONIBLES)
                derivada = sp.diff(expr, x)
                
                df_entry.delete(0, tk.END)
                df_entry.insert(0, str(derivada))
                resultado.delete('1.0', tk.END)
                resultado.insert(tk.END, f"✅ Derivada calculada automáticamente:\nf'(x) = {derivada}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo calcular la derivada:\n{e}")
        
        def calcular():
            try:
                func_str = f_entry.get()
                dfunc_str = df_entry.get()
                x0 = float(x0_entry.get())
                tol = float(tol_entry.get())
                it = int(it_entry.get())
                
                # Parsear funciones
                f, expr_f, error_f = parsear_funcion(func_str)
                if error_f:
                    resultado.delete('1.0', tk.END)
                    resultado.insert(tk.END, f"❌ Error en f(x): {error_f}")
                    return
                
                df, expr_df, error_df = parsear_funcion(dfunc_str)
                if error_df:
                    resultado.delete('1.0', tk.END)
                    resultado.insert(tk.END, f"❌ Error en f'(x): {error_df}")
                    return
                
                res, mensaje = newton_1var(f, df, x0, tol, it)
                resultado.delete('1.0', tk.END)
                if res is None:
                    resultado.insert(tk.END, f"❌ {mensaje}")
                else:
                    resultado.insert(tk.END, f"✅ Raíz aproximada: {res:.15f}\n📊 {mensaje}\n\nf(x) = {expr_f}\nf'(x) = {expr_df}")
            except ValueError as e:
                resultado.delete('1.0', tk.END)
                resultado.insert(tk.END, f"❌ Error en entrada numérica: {e}")
            except Exception as e:
                resultado.delete('1.0', tk.END)
                resultado.insert(tk.END, f"❌ Error inesperado: {e}")
        
        def limpiar():
            f_entry.delete(0, tk.END)
            df_entry.delete(0, tk.END)
            x0_entry.delete(0, tk.END)
            tol_entry.delete(0, tk.END)
            it_entry.delete(0, tk.END)
            resultado.delete('1.0', tk.END)
        
        ttk.Button(button_frame, text="Calcular Derivada", command=calcular_derivada).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Calcular", command=calcular).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Limpiar", command=limpiar).pack(side='left', padx=5)
        
        # Empacar canvas y scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def pestana_lagrange(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📉 Lagrange")
        
        # Configurar grid para centrado
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=0)
        frame.columnconfigure(2, weight=1)
        
        title = ttk.Label(frame, text="Interpolación de Lagrange", style='Title.TLabel')
        title.grid(row=0, column=1, pady=15)
        
        input_frame = ttk.LabelFrame(frame, text="Datos", padding=15)
        input_frame.grid(row=1, column=1, padx=15, pady=10, sticky='ew')
        input_frame.columnconfigure(1, weight=1)
        
        ttk.Label(input_frame, text="Valores x (separados por coma):", style='Subtitle.TLabel').grid(row=0, column=0, sticky='w', pady=8)
        ttk.Label(input_frame, text="Valores y (separados por coma):", style='Subtitle.TLabel').grid(row=1, column=0, sticky='w', pady=8)
        ttk.Label(input_frame, text="x a interpolar (opcional):", style='Subtitle.TLabel').grid(row=2, column=0, sticky='w', pady=8)
        ttk.Label(input_frame, text="Grado del polinomio (opcional):", style='Subtitle.TLabel').grid(row=3, column=0, sticky='w', pady=8)
        
        x_entry = ttk.Entry(input_frame, width=70)
        y_entry = ttk.Entry(input_frame, width=70)
        xval_entry = ttk.Entry(input_frame, width=70)
        grado_entry = ttk.Entry(input_frame, width=70)
        
        x_entry.grid(row=0, column=1, sticky='ew', padx=10)
        y_entry.grid(row=1, column=1, sticky='ew', padx=10)
        xval_entry.grid(row=2, column=1, sticky='ew', padx=10)
        grado_entry.grid(row=3, column=1, sticky='ew', padx=10)
        
        # Guardar referencias
        self.entries_por_pestana['lagrange'] = [x_entry, y_entry, xval_entry, grado_entry]
        
        result_frame = ttk.LabelFrame(frame, text="Resultado", padding=15)
        result_frame.grid(row=2, column=1, padx=15, pady=10, sticky='ew')
        
        resultado = tk.Text(result_frame, height=8, width=70, font=('Consolas', 9), bg='#F0F8FF', relief='flat')
        resultado.pack(fill='both', expand=True)
        
        # Guardar referencia
        self.resultados_por_pestana['lagrange'] = resultado
        
        def calcular():
            try:
                xs = list(map(float, x_entry.get().split(',')))
                ys = list(map(float, y_entry.get().split(',')))
                
                resultado.delete('1.0', tk.END)
                
                # Obtener la función polinómica
                func, poly, coeffs = lagrange_interp(xs, ys)
                
                resultado.insert(tk.END, "✅ Polinomio de Interpolación de Lagrange\n\n")
                
                if poly is not None:
                    # Mostrar el polinomio simbólico
                    resultado.insert(tk.END, f"📊 Función:\n  P(x) = {poly}\n\n")
                
                # Mostrar el grado del polinomio
                grado_actual = len(xs) - 1
                resultado.insert(tk.END, f"📈 Grado del polinomio: {grado_actual}\n")
                
                if coeffs:
                    resultado.insert(tk.END, f"\n📋 Coeficientes (de mayor a menor grado):\n")
                    for i, coef in enumerate(coeffs):
                        exp = grado_actual - i
                        resultado.insert(tk.END, f"  x^{exp}: {coef:.10e}\n")
                
                # Si se proporciona un x a interpolar
                xval_str = xval_entry.get().strip()
                if xval_str:
                    try:
                        xval = float(xval_str)
                        res = func(xval)
                        resultado.insert(tk.END, f"\n🔢 Valor en x = {xval}: {res:.15f}\n")
                    except:
                        pass
                
            except ValueError as e:
                resultado.delete('1.0', tk.END)
                resultado.insert(tk.END, f"❌ Error en entrada numérica: {e}")
            except Exception as e:
                resultado.delete('1.0', tk.END)
                resultado.insert(tk.END, f"❌ Error: {e}")
        
        def limpiar():
            x_entry.delete(0, tk.END)
            y_entry.delete(0, tk.END)
            xval_entry.delete(0, tk.END)
            grado_entry.delete(0, tk.END)
            resultado.delete('1.0', tk.END)
        
        # Frame para botones
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=3, column=1, pady=20)
        
        ttk.Button(button_frame, text="Calcular", command=calcular).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Limpiar", command=limpiar).pack(side='left', padx=5)

    def pestana_diferencias_divididas(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🔢 Dif. Divididas")
        
        # Configurar grid para centrado
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=0)
        frame.columnconfigure(2, weight=1)
        
        title = ttk.Label(frame, text="Diferencias Divididas de Newton", style='Title.TLabel')
        title.grid(row=0, column=1, pady=15)
        
        input_frame = ttk.LabelFrame(frame, text="Datos", padding=15)
        input_frame.grid(row=1, column=1, padx=15, pady=10, sticky='ew')
        input_frame.columnconfigure(1, weight=1)
        
        ttk.Label(input_frame, text="Valores x (separados por coma):", style='Subtitle.TLabel').grid(row=0, column=0, sticky='w', pady=8)
        ttk.Label(input_frame, text="Valores y (separados por coma):", style='Subtitle.TLabel').grid(row=1, column=0, sticky='w', pady=8)
        
        x_entry = ttk.Entry(input_frame, width=70)
        y_entry = ttk.Entry(input_frame, width=70)
        
        x_entry.grid(row=0, column=1, sticky='ew', padx=10)
        y_entry.grid(row=1, column=1, sticky='ew', padx=10)
        
        # Guardar referencias
        self.entries_por_pestana['diferencias_divididas'] = [x_entry, y_entry]
        
        result_frame = ttk.LabelFrame(frame, text="Tabla de Diferencias", padding=15)
        result_frame.grid(row=2, column=1, padx=15, pady=10, sticky='ew')
        
        resultado = tk.Text(result_frame, height=8, width=70, font=('Consolas', 9), bg='#F0F8FF', relief='flat')
        resultado.pack(fill='both', expand=True)
        
        # Guardar referencia
        self.resultados_por_pestana['diferencias_divididas'] = resultado
        
        def calcular():
            try:
                xs = list(map(float, x_entry.get().split(',')))
                ys = list(map(float, y_entry.get().split(',')))
                tabla = diferencias_divididas(xs, ys)
                n = len(xs)
                # Construir encabezado dinámico según n
                headers = ['x_i', 'f[x_i]'] + [f"{j}ª" for j in range(1, n)]
                col_width = 12
                tabla_str = "Tabla de Diferencias Divididas:\n\n"
                tabla_str += " | ".join([h.rjust(col_width) for h in headers]) + "\n"
                tabla_str += "=" * (len(headers) * (col_width + 3)) + "\n"

                for i in range(n):
                    row_items = []
                    row_items.append(f"{xs[i]:.6f}".rjust(col_width))
                    row_items.append(f"{tabla[i,0]:.8f}".rjust(col_width))
                    for j in range(1, n):
                        if j <= n - 1 - i:
                            row_items.append(f"{tabla[i,j]:.8f}".rjust(col_width))
                        else:
                            row_items.append(''.rjust(col_width))
                    tabla_str += " | ".join(row_items) + "\n"

                resultado.delete('1.0', tk.END)
                resultado.insert(tk.END, tabla_str)
            except Exception as e:
                resultado.delete('1.0', tk.END)
                resultado.insert(tk.END, f"❌ Error: {e}")
        
        def limpiar():
            x_entry.delete(0, tk.END)
            y_entry.delete(0, tk.END)
            resultado.delete('1.0', tk.END)
        
        # Frame para botones
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=3, column=1, pady=20)
        
        ttk.Button(button_frame, text="Calcular", command=calcular).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Limpiar", command=limpiar).pack(side='left', padx=5)

    def pestana_jacobi(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🔁 Jacobi / Newton")
        
        # Canvas scrollable
        canvas = tk.Canvas(frame, bg="#F5F7FA", highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollable_frame.columnconfigure(0, weight=1)
        
        title = ttk.Label(scrollable_frame, text="Método de Jacobi / Newton Multivariable", style='Title.TLabel')
        title.pack(pady=15)
        
        # Frame para selector de tipo
        type_frame = ttk.LabelFrame(scrollable_frame, text="Tipo de Sistema", padding=15)
        type_frame.pack(fill='x', padx=20, pady=10)
        
        tipo_var = tk.StringVar(value="nolineal2")
        ttk.Radiobutton(type_frame, text="Sistema Lineal (Jacobi)", 
                       variable=tipo_var, value="lineal").pack(anchor='w', pady=5)
        ttk.Radiobutton(type_frame, text="Sistema No Lineal 2 ecuaciones (Newton)", 
                       variable=tipo_var, value="nolineal2").pack(anchor='w', pady=5)
        ttk.Radiobutton(type_frame, text="Sistema No Lineal 3 ecuaciones (Newton)", 
                       variable=tipo_var, value="nolineal3").pack(anchor='w', pady=5)
        ttk.Radiobutton(type_frame, text="Sistema No Lineal 4 ecuaciones (Newton)", 
                       variable=tipo_var, value="nolineal4").pack(anchor='w', pady=5)
        
        input_frame = ttk.LabelFrame(scrollable_frame, text="Entrada de Datos", padding=15)
        input_frame.pack(fill='x', padx=20, pady=10)
        input_frame.columnconfigure(1, weight=1)
        
        # Elementos para sistema lineal
        ttk.Label(input_frame, text="Matriz A (lineal: 10,-1;-1,10):", style='Subtitle.TLabel').grid(row=0, column=0, sticky='w', pady=5)
        A_entry = ttk.Entry(input_frame, width=70)
        A_entry.grid(row=0, column=1, sticky='ew', padx=10)
        
        ttk.Label(input_frame, text="Vector b (separado por coma):", style='Subtitle.TLabel').grid(row=1, column=0, sticky='w', pady=5)
        b_entry = ttk.Entry(input_frame, width=70)
        b_entry.grid(row=1, column=1, sticky='ew', padx=10)
        
        # Elementos para sistemas no lineales
        ttk.Label(input_frame, text="Ecuación 1:", style='Subtitle.TLabel').grid(row=2, column=0, sticky='w', pady=5)
        eq1_entry = ttk.Entry(input_frame, width=70)
        eq1_entry.grid(row=2, column=1, sticky='ew', padx=10)
        
        ttk.Label(input_frame, text="Ecuación 2:", style='Subtitle.TLabel').grid(row=3, column=0, sticky='w', pady=5)
        eq2_entry = ttk.Entry(input_frame, width=70)
        eq2_entry.grid(row=3, column=1, sticky='ew', padx=10)
        
        ttk.Label(input_frame, text="Ecuación 3:", style='Subtitle.TLabel').grid(row=4, column=0, sticky='w', pady=5)
        eq3_entry = ttk.Entry(input_frame, width=70)
        eq3_entry.grid(row=4, column=1, sticky='ew', padx=10)
        
        ttk.Label(input_frame, text="Ecuación 4:", style='Subtitle.TLabel').grid(row=5, column=0, sticky='w', pady=5)
        eq4_entry = ttk.Entry(input_frame, width=70)
        eq4_entry.grid(row=5, column=1, sticky='ew', padx=10)
        
        # Estimaciones iniciales
        ttk.Label(input_frame, text="Estimación inicial (x0,y0,...):", style='Subtitle.TLabel').grid(row=6, column=0, sticky='w', pady=5)
        x0_entry = ttk.Entry(input_frame, width=70)
        x0_entry.grid(row=6, column=1, sticky='ew', padx=10)
        
        # Parámetros de Newton
        ttk.Label(input_frame, text="Tolerancia (tol):", style='Subtitle.TLabel').grid(row=7, column=0, sticky='w', pady=5)
        tol_n_entry = ttk.Entry(input_frame, width=20)
        tol_n_entry.grid(row=7, column=1, sticky='w', padx=10)
        tol_n_entry.insert(0, '1e-6')

        ttk.Label(input_frame, text="Iteraciones máx:", style='Subtitle.TLabel').grid(row=8, column=0, sticky='w', pady=5)
        maxit_n_entry = ttk.Entry(input_frame, width=20)
        maxit_n_entry.grid(row=8, column=1, sticky='w', padx=10)
        maxit_n_entry.insert(0, '50')

        ttk.Label(input_frame, text="Mostrar iteracion k (opcional):", style='Subtitle.TLabel').grid(row=9, column=0, sticky='w', pady=5)
        showk_entry = ttk.Entry(input_frame, width=20)
        showk_entry.grid(row=9, column=1, sticky='w', padx=10)
        showk_entry.insert(0, '2')
        
        result_frame = ttk.LabelFrame(scrollable_frame, text="Solución", padding=15)
        result_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        resultado = tk.Text(result_frame, height=6, width=70, font=('Consolas', 10), bg='#F0F8FF', relief='flat')
        resultado.pack(fill='both', expand=True)
        
        # Guardar referencias
        self.entries_por_pestana['jacobi'] = [A_entry, b_entry, eq1_entry, eq2_entry, eq3_entry, eq4_entry, x0_entry]
        self.resultados_por_pestana['jacobi'] = resultado
        
        def calcular():
            try:
                resultado.delete('1.0', tk.END)
                
                if tipo_var.get() == "lineal":
                    # Sistema lineal con Jacobi
                    if not A_entry.get().strip() or not b_entry.get().strip() or not x0_entry.get().strip():
                        resultado.insert(tk.END, "❌ Completa 'Matriz A', 'Vector b' y 'Estimación inicial'\n")
                        return
                    A = np.array([list(map(float, fila.split(','))) for fila in A_entry.get().split(';')])
                    b = np.array(list(map(float, b_entry.get().split(','))))
                    x0 = np.array(list(map(float, x0_entry.get().split(','))))
                    
                    res = jacobi(A, b, x0)
                    resultado.insert(tk.END, f"✅ Sistema Lineal - Método de Jacobi\n\n")
                    resultado.insert(tk.END, f"Solución:\n")
                    for i, val in enumerate(res):
                        resultado.insert(tk.END, f"  Variable {i}: {val:.15f}\n")
                
                elif tipo_var.get() == "nolineal2":
                    # Sistema no lineal 2 ecuaciones
                    eq1 = eq1_entry.get().strip()
                    eq2 = eq2_entry.get().strip()
                    
                    if not eq1 or not eq2:
                        resultado.insert(tk.END, "❌ Ingresa las 2 ecuaciones")
                        return
                    
                    # validar estimaciones iniciales
                    x0_text = x0_entry.get().strip()
                    if not x0_text:
                        resultado.insert(tk.END, "❌ Ingresa al menos 2 estimaciones iniciales (x0,y0)\n")
                        return
                    x0_vals = list(map(float, x0_text.split(',')))
                    if len(x0_vals) < 2:
                        resultado.insert(tk.END, "❌ Ingresa al menos 2 estimaciones iniciales (x0,y0)")
                        return
                    
                    # Usar Newton multivariable localmente para obtener historial
                    tol = float(tol_n_entry.get()) if tol_n_entry.get().strip() else 1e-6
                    maxit = int(maxit_n_entry.get()) if maxit_n_entry.get().strip() else 50
                    show_k = int(showk_entry.get()) if showk_entry.get().strip() else 2

                    try:
                        x_sym, y_sym = sp.symbols('x y')
                        f1_expr = sp.sympify(normalizar_ecuacion_str(eq1), locals=FUNCIONES_DISPONIBLES)
                        f2_expr = sp.sympify(normalizar_ecuacion_str(eq2), locals=FUNCIONES_DISPONIBLES)
                        J = sp.Matrix([[sp.diff(f1_expr, x_sym), sp.diff(f1_expr, y_sym)],
                                       [sp.diff(f2_expr, x_sym), sp.diff(f2_expr, y_sym)]])

                        f1 = sp.lambdify((x_sym, y_sym), f1_expr, 'numpy')
                        f2 = sp.lambdify((x_sym, y_sym), f2_expr, 'numpy')
                        Jf = sp.lambdify((x_sym, y_sym), J, 'numpy')

                        xk = np.array([x0_vals[0], x0_vals[1]], dtype=float)
                        history = [xk.copy()]
                        msg = ''
                        for it in range(1, maxit+1):
                            Fk = np.array([f1(xk[0], xk[1]), f2(xk[0], xk[1])], dtype=float)
                            Jk = Jf(xk[0], xk[1]).astype(float)
                            try:
                                delta = np.linalg.solve(Jk, -Fk)
                            except np.linalg.LinAlgError:
                                msg = 'Matriz jacobiana singular'
                                break
                            xk = xk + delta
                            history.append(xk.copy())
                            if np.linalg.norm(delta) < tol:
                                msg = f'Convergió en {it} iteraciones'
                                break
                        else:
                            msg = f'Máximo de iteraciones alcanzado ({maxit})'

                        if history:
                            resultado.insert(tk.END, f"✅ Sistema No Lineal 2 Ecuaciones - Newton\n")
                            resultado.insert(tk.END, f"📊 {msg}\n\n")
                            resultado.insert(tk.END, f"Ecuación 1: {eq1} = 0\n")
                            resultado.insert(tk.END, f"Ecuación 2: {eq2} = 0\n\n")
                            # Mostrar X^(k) solicitado si existe
                            if show_k < len(history):
                                xk_show = history[show_k]
                                resultado.insert(tk.END, f"X^({show_k}) = [{xk_show[0]:.15f}, {xk_show[1]:.15f}]\n\n")
                            # Mostrar siempre la última aproximación
                            xk_final = history[-1]
                            resultado.insert(tk.END, f"Solución aproximada (última):\n  x = {xk_final[0]:.15f}\n  y = {xk_final[1]:.15f}\n")
                            # Mostrar norma del residual F(x)
                            try:
                                Fvals = np.array([f1(xk_final[0], xk_final[1]), f2(xk_final[0], xk_final[1])], dtype=float)
                                resultado.insert(tk.END, f"||F(X)|| = {np.linalg.norm(Fvals):.6e}\n")
                            except Exception:
                                pass
                    except Exception as e:
                        resultado.insert(tk.END, f"❌ Error al ejecutar Newton: {e}")
                
                elif tipo_var.get() == "nolineal3":
                    # Sistema no lineal 3 ecuaciones
                    eq1 = eq1_entry.get().strip()
                    eq2 = eq2_entry.get().strip()
                    eq3 = eq3_entry.get().strip()
                    
                    if not eq1 or not eq2 or not eq3:
                        resultado.insert(tk.END, "❌ Ingresa las 3 ecuaciones")
                        return
                    
                    x0_text = x0_entry.get().strip()
                    if not x0_text:
                        resultado.insert(tk.END, "❌ Ingresa al menos 3 estimaciones iniciales (x0,y0,z0)\n")
                        return
                    x0_vals = list(map(float, x0_text.split(',')))
                    if len(x0_vals) < 3:
                        resultado.insert(tk.END, "❌ Ingresa al menos 3 estimaciones iniciales (x0,y0,z0)")
                        return
                    
                    # Newton multivariable para 3 ecuaciones
                    tol = float(tol_n_entry.get()) if tol_n_entry.get().strip() else 1e-6
                    maxit = int(maxit_n_entry.get()) if maxit_n_entry.get().strip() else 50
                    show_k = int(showk_entry.get()) if showk_entry.get().strip() else 2
                    try:
                        x_sym, y_sym, z_sym = sp.symbols('x y z')
                        f1_expr = sp.sympify(normalizar_ecuacion_str(eq1), locals=FUNCIONES_DISPONIBLES)
                        f2_expr = sp.sympify(normalizar_ecuacion_str(eq2), locals=FUNCIONES_DISPONIBLES)
                        f3_expr = sp.sympify(normalizar_ecuacion_str(eq3), locals=FUNCIONES_DISPONIBLES)
                        J = sp.Matrix([
                            [sp.diff(f1_expr, x_sym), sp.diff(f1_expr, y_sym), sp.diff(f1_expr, z_sym)],
                            [sp.diff(f2_expr, x_sym), sp.diff(f2_expr, y_sym), sp.diff(f2_expr, z_sym)],
                            [sp.diff(f3_expr, x_sym), sp.diff(f3_expr, y_sym), sp.diff(f3_expr, z_sym)]
                        ])

                        f1 = sp.lambdify((x_sym, y_sym, z_sym), f1_expr, 'numpy')
                        f2 = sp.lambdify((x_sym, y_sym, z_sym), f2_expr, 'numpy')
                        f3 = sp.lambdify((x_sym, y_sym, z_sym), f3_expr, 'numpy')
                        Jf = sp.lambdify((x_sym, y_sym, z_sym), J, 'numpy')

                        xk = np.array([x0_vals[0], x0_vals[1], x0_vals[2]], dtype=float)
                        history = [xk.copy()]
                        msg = ''
                        for it in range(1, maxit+1):
                            Fk = np.array([f1(xk[0], xk[1], xk[2]), f2(xk[0], xk[1], xk[2]), f3(xk[0], xk[1], xk[2])], dtype=float)
                            Jk = Jf(xk[0], xk[1], xk[2]).astype(float)
                            try:
                                delta = np.linalg.solve(Jk, -Fk)
                            except np.linalg.LinAlgError:
                                msg = 'Matriz jacobiana singular'
                                break
                            xk = xk + delta
                            history.append(xk.copy())
                            if np.linalg.norm(delta) < tol:
                                msg = f'Convergió en {it} iteraciones'
                                break
                        else:
                            msg = f'Máximo de iteraciones alcanzado ({maxit})'

                        resultado.insert(tk.END, f"✅ Sistema No Lineal 3 Ecuaciones - Newton\n")
                        resultado.insert(tk.END, f"📊 {msg}\n\n")
                        resultado.insert(tk.END, f"Ecuación 1: {eq1} = 0\n")
                        resultado.insert(tk.END, f"Ecuación 2: {eq2} = 0\n")
                        resultado.insert(tk.END, f"Ecuación 3: {eq3} = 0\n\n")
                        if show_k < len(history):
                            xk_show = history[show_k]
                            resultado.insert(tk.END, f"X^({show_k}) = [{', '.join(f'{v:.15f}' for v in xk_show)}]\n\n")
                        xk_final = history[-1]
                        resultado.insert(tk.END, f"Solución aproximada (última):\n  x = {xk_final[0]:.15f}\n  y = {xk_final[1]:.15f}\n  z = {xk_final[2]:.15f}\n")
                    except Exception as e:
                        resultado.insert(tk.END, f"❌ Error al ejecutar Newton: {e}")
                
                elif tipo_var.get() == "nolineal4":
                    # Sistema no lineal 4 ecuaciones
                    eq1 = eq1_entry.get().strip()
                    eq2 = eq2_entry.get().strip()
                    eq3 = eq3_entry.get().strip()
                    eq4 = eq4_entry.get().strip()
                    
                    if not eq1 or not eq2 or not eq3 or not eq4:
                        resultado.insert(tk.END, "❌ Ingresa las 4 ecuaciones")
                        return
                    
                    x0_text = x0_entry.get().strip()
                    if not x0_text:
                        resultado.insert(tk.END, "❌ Ingresa al menos 4 estimaciones iniciales (x0,y0,z0,w0)\n")
                        return
                    x0_vals = list(map(float, x0_text.split(',')))
                    if len(x0_vals) < 4:
                        resultado.insert(tk.END, "❌ Ingresa al menos 4 estimaciones iniciales (x0,y0,z0,w0)")
                        return
                    
                    res, it, msg = resolver_sistema_newton_4(eq1, eq2, eq3, eq4, x0_vals[0], x0_vals[1], x0_vals[2], x0_vals[3])
                    
                    if res is None:
                        resultado.insert(tk.END, f"❌ {msg}")
                    else:
                        resultado.insert(tk.END, f"✅ Sistema No Lineal 4 Ecuaciones - Newton\n")
                        resultado.insert(tk.END, f"📊 {msg}\n\n")
                        resultado.insert(tk.END, f"Ecuación 1: {eq1} = 0\n")
                        resultado.insert(tk.END, f"Ecuación 2: {eq2} = 0\n")
                        resultado.insert(tk.END, f"Ecuación 3: {eq3} = 0\n")
                        resultado.insert(tk.END, f"Ecuación 4: {eq4} = 0\n\n")
                        resultado.insert(tk.END, f"Solución:\n")
                        resultado.insert(tk.END, f"  x = {res[0]:.15f}\n")
                        resultado.insert(tk.END, f"  y = {res[1]:.15f}\n")
                        resultado.insert(tk.END, f"  z = {res[2]:.15f}\n")
                        resultado.insert(tk.END, f"  w = {res[3]:.15f}\n")
            
            except Exception as e:
                resultado.delete('1.0', tk.END)
                resultado.insert(tk.END, f"❌ Error: {e}\n\nVerifica:\n- Formato de entrada correcto\n- Estimaciones iniciales válidas\n- Ecuaciones con sintaxis correcta")
        
        def limpiar():
            A_entry.delete(0, tk.END)
            b_entry.delete(0, tk.END)
            eq1_entry.delete(0, tk.END)
            eq2_entry.delete(0, tk.END)
            eq3_entry.delete(0, tk.END)
            eq4_entry.delete(0, tk.END)
            x0_entry.delete(0, tk.END)
            resultado.delete('1.0', tk.END)
        
        # Frame para botones
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Calcular", command=calcular).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Limpiar", command=limpiar).pack(side='left', padx=5)
        
        # Empacar canvas y scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def pestana_gauss_seidel(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🔄 Gauss-Seidel")
        
        # Canvas scrollable
        canvas = tk.Canvas(frame, bg="#F5F7FA", highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollable_frame.columnconfigure(0, weight=1)
        
        title = ttk.Label(scrollable_frame, text="Gauss-Seidel / Newton Multivariable", style='Title.TLabel')
        title.pack(pady=15)
        
        # Frame para selector de tipo
        type_frame = ttk.LabelFrame(scrollable_frame, text="Tipo de Sistema", padding=15)
        type_frame.pack(fill='x', padx=20, pady=10)
        
        tipo_var = tk.StringVar(value="lineal")
        ttk.Radiobutton(type_frame, text="Sistema Lineal (Gauss-Seidel)", 
                       variable=tipo_var, value="lineal").pack(anchor='w', pady=5)
        ttk.Radiobutton(type_frame, text="Sistema No Lineal 2 ecuaciones (Newton)", 
                       variable=tipo_var, value="nolineal2").pack(anchor='w', pady=5)
        ttk.Radiobutton(type_frame, text="Sistema No Lineal 3 ecuaciones (Newton)", 
                       variable=tipo_var, value="nolineal3").pack(anchor='w', pady=5)
        ttk.Radiobutton(type_frame, text="Sistema No Lineal 4 ecuaciones (Newton)", 
                       variable=tipo_var, value="nolineal4").pack(anchor='w', pady=5)
        
        input_frame = ttk.LabelFrame(scrollable_frame, text="Entrada de Datos", padding=15)
        input_frame.pack(fill='x', padx=20, pady=10)
        input_frame.columnconfigure(1, weight=1)
        
        # Elementos para sistema lineal
        ttk.Label(input_frame, text="Matriz A (lineal: 10,-1;-1,10):", style='Subtitle.TLabel').grid(row=0, column=0, sticky='w', pady=5)
        A_entry = ttk.Entry(input_frame, width=70)
        A_entry.grid(row=0, column=1, sticky='ew', padx=10)
        
        ttk.Label(input_frame, text="Vector b (separado por coma):", style='Subtitle.TLabel').grid(row=1, column=0, sticky='w', pady=5)
        b_entry = ttk.Entry(input_frame, width=70)
        b_entry.grid(row=1, column=1, sticky='ew', padx=10)
        
        # Elementos para sistemas no lineales
        ttk.Label(input_frame, text="Ecuación 1:", style='Subtitle.TLabel').grid(row=2, column=0, sticky='w', pady=5)
        eq1_entry = ttk.Entry(input_frame, width=70)
        eq1_entry.grid(row=2, column=1, sticky='ew', padx=10)
        
        ttk.Label(input_frame, text="Ecuación 2:", style='Subtitle.TLabel').grid(row=3, column=0, sticky='w', pady=5)
        eq2_entry = ttk.Entry(input_frame, width=70)
        eq2_entry.grid(row=3, column=1, sticky='ew', padx=10)
        
        ttk.Label(input_frame, text="Ecuación 3:", style='Subtitle.TLabel').grid(row=4, column=0, sticky='w', pady=5)
        eq3_entry = ttk.Entry(input_frame, width=70)
        eq3_entry.grid(row=4, column=1, sticky='ew', padx=10)
        
        ttk.Label(input_frame, text="Ecuación 4:", style='Subtitle.TLabel').grid(row=5, column=0, sticky='w', pady=5)
        eq4_entry = ttk.Entry(input_frame, width=70)
        eq4_entry.grid(row=5, column=1, sticky='ew', padx=10)
        
        # Estimaciones iniciales
        ttk.Label(input_frame, text="Estimación inicial (x0,y0,...):", style='Subtitle.TLabel').grid(row=6, column=0, sticky='w', pady=5)
        x0_entry = ttk.Entry(input_frame, width=70)
        x0_entry.grid(row=6, column=1, sticky='ew', padx=10)
        
        result_frame = ttk.LabelFrame(scrollable_frame, text="Solución", padding=15)
        result_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        resultado = tk.Text(result_frame, height=6, width=70, font=('Consolas', 10), bg='#F0F8FF', relief='flat')
        resultado.pack(fill='both', expand=True)
        
        # Guardar referencias
        self.entries_por_pestana['gauss_seidel'] = [A_entry, b_entry, eq1_entry, eq2_entry, eq3_entry, eq4_entry, x0_entry]
        self.resultados_por_pestana['gauss_seidel'] = resultado
        
        def calcular():
            try:
                resultado.delete('1.0', tk.END)
                
                if tipo_var.get() == "lineal":
                    # Sistema lineal con Gauss-Seidel
                    A_text = A_entry.get().strip()
                    b_text = b_entry.get().strip()
                    x0_text = x0_entry.get().strip()

                    if not b_text:
                        resultado.insert(tk.END, "❌ Completa 'Vector b'\n")
                        return

                    b_vals = list(map(float, b_text.split(',')))

                    if A_text:
                        # Parsear A directamente
                        A = np.array([list(map(float, fila.split(','))) for fila in A_text.split(';')])
                    else:
                        eqs = []
                        for e in [eq1_entry.get().strip(), eq2_entry.get().strip(), eq3_entry.get().strip(), eq4_entry.get().strip()]:
                            if e:
                                eqs.append(normalizar_ecuacion_str(e))
                        n = len(b_vals)
                        if len(eqs) < n:
                            resultado.insert(tk.END, "❌ Proporciona al menos tantas ecuaciones como entradas en b o completa Matriz A\n")
                            return
                        # Variables: x,y,z,w según tamaño
                        var_syms = list(sp.symbols('x y z w'))[:n]
                        A_rows = []
                        b_adj = []
                        try:
                            for i in range(n):
                                expr = sp.sympify(eqs[i], locals=FUNCIONES_DISPONIBLES)
                                expr = sp.expand(expr)
                                const = float(expr.subs({v:0 for v in var_syms}))
                                row = [float(sp.N(expr.coeff(v))) for v in var_syms]
                                A_rows.append(row)
                                b_adj.append(b_vals[i] - const)
                            A = np.array(A_rows, dtype=float)
                            b_vals = np.array(b_adj, dtype=float)
                        except Exception as ex:
                            resultado.insert(tk.END, f"❌ Error al construir A desde ecuaciones: {ex}\n")
                            return

                    if not x0_text:
                        resultado.insert(tk.END, "❌ Completa 'Estimación inicial'\n")
                        return
                    x0 = np.array(list(map(float, x0_text.split(','))))

                    res = gauss_seidel(A, np.array(b_vals), x0)
                    resultado.insert(tk.END, f"✅ Sistema Lineal - Método de Gauss-Seidel\n\n")
                    resultado.insert(tk.END, f"Solución:\n")
                    for i, val in enumerate(res):
                        resultado.insert(tk.END, f"  Variable {i}: {val:.15f}\n")
                
                elif tipo_var.get() == "nolineal2":
                    # Sistema no lineal 2 ecuaciones
                    eq1 = eq1_entry.get().strip()
                    eq2 = eq2_entry.get().strip()
                    
                    if not eq1 or not eq2:
                        resultado.insert(tk.END, "❌ Ingresa las 2 ecuaciones")
                        return
                    
                    x0_vals = list(map(float, x0_entry.get().split(',')))
                    if len(x0_vals) < 2:
                        resultado.insert(tk.END, "❌ Ingresa al menos 2 estimaciones iniciales (x0,y0)")
                        return
                    
                    eq1_norm = normalizar_ecuacion_str(eq1)
                    eq2_norm = normalizar_ecuacion_str(eq2)
                    res, it, msg = resolver_sistema_newton_2(eq1_norm, eq2_norm, x0_vals[0], x0_vals[1])
                    
                    if res is None:
                        resultado.insert(tk.END, f"❌ {msg}")
                    else:
                        resultado.insert(tk.END, f"✅ Sistema No Lineal 2 Ecuaciones - Newton\n")
                        resultado.insert(tk.END, f"📊 {msg}\n\n")
                        resultado.insert(tk.END, f"Ecuación 1: {eq1} = 0\n")
                        resultado.insert(tk.END, f"Ecuación 2: {eq2} = 0\n\n")
                        resultado.insert(tk.END, f"Solución:\n")
                        resultado.insert(tk.END, f"  x = {res[0]:.15f}\n")
                        resultado.insert(tk.END, f"  y = {res[1]:.15f}\n")
                        # calcular norma residual si es posible
                        try:
                            x_sym, y_sym = sp.symbols('x y')
                            f1_expr = sp.sympify(eq1_norm, locals=FUNCIONES_DISPONIBLES)
                            f2_expr = sp.sympify(eq2_norm, locals=FUNCIONES_DISPONIBLES)
                            f1f = sp.lambdify((x_sym, y_sym), f1_expr, 'numpy')
                            f2f = sp.lambdify((x_sym, y_sym), f2_expr, 'numpy')
                            Fvals = np.array([f1f(res[0], res[1]), f2f(res[0], res[1])], dtype=float)
                            resultado.insert(tk.END, f"||F(X)|| = {np.linalg.norm(Fvals):.6e}\n")
                        except Exception:
                            pass
                
                elif tipo_var.get() == "nolineal3":
                    # Sistema no lineal 3 ecuaciones
                    eq1 = eq1_entry.get().strip()
                    eq2 = eq2_entry.get().strip()
                    eq3 = eq3_entry.get().strip()
                    
                    if not eq1 or not eq2 or not eq3:
                        resultado.insert(tk.END, "❌ Ingresa las 3 ecuaciones")
                        return
                    
                    x0_vals = list(map(float, x0_entry.get().split(',')))
                    if len(x0_vals) < 3:
                        resultado.insert(tk.END, "❌ Ingresa al menos 3 estimaciones iniciales (x0,y0,z0)")
                        return
                    
                    res, it, msg = resolver_sistema_newton_3(eq1, eq2, eq3, x0_vals[0], x0_vals[1], x0_vals[2])
                    
                    if res is None:
                        resultado.insert(tk.END, f"❌ {msg}")
                    else:
                        resultado.insert(tk.END, f"✅ Sistema No Lineal 3 Ecuaciones - Newton\n")
                        resultado.insert(tk.END, f"📊 {msg}\n\n")
                        resultado.insert(tk.END, f"Ecuación 1: {eq1} = 0\n")
                        resultado.insert(tk.END, f"Ecuación 2: {eq2} = 0\n")
                        resultado.insert(tk.END, f"Ecuación 3: {eq3} = 0\n\n")
                        resultado.insert(tk.END, f"Solución:\n")
                        resultado.insert(tk.END, f"  x = {res[0]:.15f}\n")
                        resultado.insert(tk.END, f"  y = {res[1]:.15f}\n")
                        resultado.insert(tk.END, f"  z = {res[2]:.15f}\n")
                
                elif tipo_var.get() == "nolineal4":
                    # Sistema no lineal 4 ecuaciones
                    eq1 = eq1_entry.get().strip()
                    eq2 = eq2_entry.get().strip()
                    eq3 = eq3_entry.get().strip()
                    eq4 = eq4_entry.get().strip()
                    
                    if not eq1 or not eq2 or not eq3 or not eq4:
                        resultado.insert(tk.END, "❌ Ingresa las 4 ecuaciones")
                        return
                    
                    x0_vals = list(map(float, x0_entry.get().split(',')))
                    if len(x0_vals) < 4:
                        resultado.insert(tk.END, "❌ Ingresa al menos 4 estimaciones iniciales (x0,y0,z0,w0)")
                        return
                    
                    res, it, msg = resolver_sistema_newton_4(eq1, eq2, eq3, eq4, x0_vals[0], x0_vals[1], x0_vals[2], x0_vals[3])
                    
                    if res is None:
                        resultado.insert(tk.END, f"❌ {msg}")
                    else:
                        resultado.insert(tk.END, f"✅ Sistema No Lineal 4 Ecuaciones - Newton\n")
                        resultado.insert(tk.END, f"📊 {msg}\n\n")
                        resultado.insert(tk.END, f"Ecuación 1: {eq1} = 0\n")
                        resultado.insert(tk.END, f"Ecuación 2: {eq2} = 0\n")
                        resultado.insert(tk.END, f"Ecuación 3: {eq3} = 0\n")
                        resultado.insert(tk.END, f"Ecuación 4: {eq4} = 0\n\n")
                        resultado.insert(tk.END, f"Solución:\n")
                        resultado.insert(tk.END, f"  x = {res[0]:.15f}\n")
                        resultado.insert(tk.END, f"  y = {res[1]:.15f}\n")
                        resultado.insert(tk.END, f"  z = {res[2]:.15f}\n")
                        resultado.insert(tk.END, f"  w = {res[3]:.15f}\n")
            
            except Exception as e:
                resultado.delete('1.0', tk.END)
                resultado.insert(tk.END, f"❌ Error: {e}\n\nVerifica:\n- Formato de entrada correcto\n- Estimaciones iniciales válidas\n- Ecuaciones con sintaxis correcta")
        
        def limpiar():
            A_entry.delete(0, tk.END)
            b_entry.delete(0, tk.END)
            eq1_entry.delete(0, tk.END)
            eq2_entry.delete(0, tk.END)
            eq3_entry.delete(0, tk.END)
            eq4_entry.delete(0, tk.END)
            x0_entry.delete(0, tk.END)
            resultado.delete('1.0', tk.END)
        
        # Frame para botones
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Calcular", command=calcular).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Limpiar", command=limpiar).pack(side='left', padx=5)
        
        # Empacar canvas y scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def pestana_minimos_cuadrados(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📐 Mínimos Cuadrados")
        
        # Configurar grid para centrado
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=0)
        frame.columnconfigure(2, weight=1)
        
        title = ttk.Label(frame, text="Ajuste por Mínimos Cuadrados", style='Title.TLabel')
        title.grid(row=0, column=1, pady=15)
        
        input_frame = ttk.LabelFrame(frame, text="Datos", padding=15)
        input_frame.grid(row=1, column=1, padx=15, pady=10, sticky='ew')
        input_frame.columnconfigure(1, weight=1)
        
        ttk.Label(input_frame, text="Valores x (separados por coma):", style='Subtitle.TLabel').grid(row=0, column=0, sticky='w', pady=8)
        ttk.Label(input_frame, text="Valores y (separados por coma):", style='Subtitle.TLabel').grid(row=1, column=0, sticky='w', pady=8)
        
        x_entry = ttk.Entry(input_frame, width=70)
        y_entry = ttk.Entry(input_frame, width=70)
        
        x_entry.grid(row=0, column=1, sticky='ew', padx=10)
        y_entry.grid(row=1, column=1, sticky='ew', padx=10)
        
        # Guardar referencias
        self.entries_por_pestana['minimos_cuadrados'] = [x_entry, y_entry]
        
        result_frame = ttk.LabelFrame(frame, text="Ecuación Ajustada", padding=15)
        result_frame.grid(row=2, column=1, padx=15, pady=10, sticky='ew')
        
        resultado = tk.Text(result_frame, height=4, width=70, font=('Consolas', 10), bg='#F0F8FF', relief='flat')
        resultado.pack(fill='both', expand=True)
        
        # Guardar referencia
        self.resultados_por_pestana['minimos_cuadrados'] = resultado
        
        def calcular():
            try:
                xs = list(map(float, x_entry.get().split(',')))
                ys = list(map(float, y_entry.get().split(',')))
                m, b = minimos_cuadrados(xs, ys)
                resultado.delete('1.0', tk.END)
                resultado.insert(tk.END, f"✅ Ecuación de regresión lineal:\ny = {m:.10f}x + {b:.10f}\n\nPendiente (m): {m}\nIntercepto (b): {b}")
            except Exception as e:
                resultado.delete('1.0', tk.END)
                resultado.insert(tk.END, f"❌ Error: {e}")
        
        def limpiar():
            x_entry.delete(0, tk.END)
            y_entry.delete(0, tk.END)
            resultado.delete('1.0', tk.END)
        
        # Frame para botones
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=3, column=1, pady=20)
        
        ttk.Button(button_frame, text="Calcular", command=calcular).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Limpiar", command=limpiar).pack(side='left', padx=5)

    def pestana_diferencias_finitas(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📐 Dif. Finitas")

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=0)
        frame.columnconfigure(2, weight=1)

        title = ttk.Label(frame, text="Derivación por Diferencias Finitas", style='Title.TLabel')
        title.grid(row=0, column=1, pady=15)

        input_frame = ttk.LabelFrame(frame, text="Parámetros", padding=15)
        input_frame.grid(row=1, column=1, padx=15, pady=10, sticky='ew')
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Función f(x):").grid(row=0, column=0, sticky='w')
        ttk.Label(input_frame, text="Límite a:").grid(row=1, column=0, sticky='w')
        ttk.Label(input_frame, text="Límite b:").grid(row=2, column=0, sticky='w')
        ttk.Label(input_frame, text="Subintervalos n:").grid(row=3, column=0, sticky='w')
        ttk.Label(input_frame, text="Tipo de diferencia:").grid(row=4, column=0, sticky='w')

        f_entry = ttk.Entry(input_frame, width=50)
        a_entry = ttk.Entry(input_frame)
        b_entry = ttk.Entry(input_frame)
        n_entry = ttk.Entry(input_frame)

        tipo_var = tk.StringVar(value='central')
        tipo_menu = ttk.Combobox(
            input_frame,
            textvariable=tipo_var,
            values=['progresiva', 'regresiva', 'central'],
            state='readonly'
        )

        f_entry.grid(row=0, column=1, padx=10, pady=5)
        a_entry.grid(row=1, column=1, padx=10, pady=5)
        b_entry.grid(row=2, column=1, padx=10, pady=5)
        n_entry.grid(row=3, column=1, padx=10, pady=5)
        tipo_menu.grid(row=4, column=1, padx=10, pady=5)

        result_frame = ttk.LabelFrame(frame, text="Tabla de resultados", padding=15)
        result_frame.grid(row=2, column=1, padx=15, pady=10, sticky='ew')

        resultado = tk.Text(result_frame, height=12, width=75, font=('Consolas', 10), bg='#F0F8FF')
        resultado.pack(fill='both', expand=True)

        def calcular():
            try:
                f, _, err = parsear_funcion(f_entry.get())
                if err:
                    raise ValueError(err)

                a = float(a_entry.get())
                b = float(b_entry.get())
                n = int(n_entry.get())
                tipo = tipo_var.get()

                x, y, xdf, df, h = diferencias_finitas(f, a, b, n, tipo)

                resultado.delete('1.0', tk.END)
                resultado.insert(tk.END, f"h = {h:.6f}\n\n")
                resultado.insert(tk.END, "   i      x_i        f(x_i)       f'(x_i)\n")
                resultado.insert(tk.END, "-" * 55 + "\n")

                for i in range(len(xdf)):
                    resultado.insert(
                        tk.END,
                        f"{i:4d}  {xdf[i]:10.6f}  {y[i+1]:12.6f}  {df[i]:12.6f}\n"
                    )

            except Exception as e:
                resultado.delete('1.0', tk.END)
                resultado.insert(tk.END, f"❌ Error: {e}")

        ttk.Button(frame, text="Calcular", command=calcular).grid(row=3, column=1, pady=10)

    
    def pestana_taylor(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📘 Taylor")

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=0)
        frame.columnconfigure(2, weight=1)

        title = ttk.Label(frame, text="Polinomio de Taylor", style='Title.TLabel')
        title.grid(row=0, column=1, pady=15)

        input_frame = ttk.LabelFrame(frame, text="Parámetros", padding=15)
        input_frame.grid(row=1, column=1, padx=15, pady=10, sticky='ew')
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Función f(x):").grid(row=0, column=0, sticky='w')
        ttk.Label(input_frame, text="Punto a:").grid(row=1, column=0, sticky='w')
        ttk.Label(input_frame, text="Grado n:").grid(row=2, column=0, sticky='w')

        f_entry = ttk.Entry(input_frame, width=50)
        a_entry = ttk.Entry(input_frame)
        n_entry = ttk.Entry(input_frame)

        f_entry.grid(row=0, column=1, padx=10, pady=5)
        a_entry.grid(row=1, column=1, padx=10, pady=5)
        n_entry.grid(row=2, column=1, padx=10, pady=5)

        result_frame = ttk.LabelFrame(frame, text="Resultados", padding=15)
        result_frame.grid(row=2, column=1, padx=15, pady=10, sticky='ew')

        resultado = tk.Text(
            result_frame,
            height=14,
            width=80,
            font=('Consolas', 10),
            bg='#F7F9FC'
        )
        resultado.pack(fill='both', expand=True)

        def calcular_taylor():
            try:
                func_str = f_entry.get()
                a = float(a_entry.get())
                n = int(n_entry.get())

                poly, _, coef, err = polinomio_taylor(func_str, a, n)
                if err:
                    raise ValueError(err)

                resultado.delete('1.0', tk.END)
                resultado.insert(tk.END, "📘 POLINOMIO DE TAYLOR\n\n")
                resultado.insert(tk.END, f"Función: f(x) = {func_str}\n")
                resultado.insert(tk.END, f"Punto de expansión: a = {a}\n")
                resultado.insert(tk.END, f"Grado: n = {n}\n\n")

                resultado.insert(tk.END, "Polinomio:\n")
                resultado.insert(tk.END, str(poly) + "\n\n")

                resultado.insert(tk.END, "Coeficientes:\n")
                resultado.insert(tk.END, "-" * 40 + "\n")

                for i, c in enumerate(coef):
                    resultado.insert(tk.END, f"a{i} = {c}\n")

            except Exception as e:
                resultado.delete('1.0', tk.END)
                resultado.insert(tk.END, f"❌ Error: {e}")

        ttk.Button(frame, text="Calcular Taylor", command=calcular_taylor)\
            .grid(row=3, column=1, pady=10)

if __name__ == "__main__":
    app = App()
    app.mainloop()