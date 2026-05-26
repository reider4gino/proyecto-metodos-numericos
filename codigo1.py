import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import lagrange, CubicSpline
from scipy.optimize import brentq
import warnings

# Ignorar advertencias genéricas
warnings.simplefilter('ignore')

class AppIntegrada:
    def __init__(self, root):
        self.root = root
        self.root.title("Software de Métodos Numéricos - Examen Parcial")
        self.root.geometry("1350x800")
        
        style = ttk.Style()
        style.theme_use('clam')
        
        # --- SISTEMA DE PESTAÑAS (TABS) ---
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab1, text=" 📊 Ejercicio 1: Interfaz de Splines Original ")
        self.notebook.add(self.tab2, text=" 📡 Ejercicio 2: Telemetría (Dashboard Avanzado) ")
        
        # Construir ambas interfaces
        self.construir_ejercicio_1(self.tab1)
        self.construir_ejercicio_2(self.tab2)

    # =================================================================================
    # PESTAÑA 1: TU CÓDIGO ORIGINAL INTACTO (RESTAURADO 100%)
    # =================================================================================
    def construir_ejercicio_1(self, parent):
        self.presets_ejes = {
            "General (X vs Y)": ("Variable Independiente (X)", "Variable Dependiente (Y)"),
            "Biomédica (Frecuencia vs Impedancia)": ("Frecuencia f (Hz)", "Magnitud de Impedancia |Z| (Ohm)"),
            "Cinemática (Tiempo vs Posición)": ("Tiempo t (s)", "Posición x (m)"),
            "Termodinámica (Temperatura vs Presión)": ("Temperatura T (°C)", "Presión P (kPa)")
        }
        
        panel_datos = ttk.LabelFrame(parent, text=" 1. Base de Datos del Experimento ", padding=10)
        panel_datos.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10, expand=False)
        
        ttk.Label(panel_datos, text="Formato: X, Y\n(Una medición por cada línea):", font=('Arial', 9, 'italic')).pack(anchor=tk.W, pady=2)
        self.txt_datos_1 = tk.Text(panel_datos, width=32, height=25, font=('Consolas', 10))
        self.txt_datos_1.pack(fill=tk.BOTH, expand=True, pady=5)
        
        datos_iniciales = (
            "100, 152.3\n120, 149.1\n145, 146.8\n170, 144.9\n200, 142.0\n"
            "235, 139.5\n270, 137.9\n310, 136.1\n355, 134.8\n405, 133.6\n"
            "460, 132.7\n520, 131.9\n585, 131.4\n655, 131.1\n730, 130.9\n"
            "810, 131.0\n895, 131.3\n985, 131.9\n1080, 132.7\n1180, 133.8\n"
            "1290, 135.2\n1410, 136.9\n1540, 138.9\n1680, 141.1\n1830, 143.5\n"
            "1990, 146.1\n2160, 149.0\n2340, 152.2\n2530, 155.6\n2730, 159.2"
        )
        self.txt_datos_1.insert(tk.END, datos_iniciales)
        
        panel_controles = ttk.LabelFrame(parent, text=" 2. Configuración y Cálculos ", padding=10)
        panel_controles.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=10)
        
        ttk.Label(panel_controles, text="Contexto / Tipo de Estudio:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=2)
        self.combo_preset = ttk.Combobox(panel_controles, values=list(self.presets_ejes.keys()), state="readonly", width=35)
        self.combo_preset.set("Biomédica (Frecuencia vs Impedancia)") 
        self.combo_preset.pack(anchor=tk.W, pady=5)
        self.combo_preset.bind("<<ComboboxSelected>>", lambda e: self.actualizar_nombres_ejes_1())
        
        ttk.Separator(panel_controles, orient='horizontal').pack(fill=tk.X, pady=10)
        
        self.lbl_eval_x_1 = ttk.Label(panel_controles, text="Evaluar en X:", font=('Arial', 10, 'bold'))
        self.lbl_eval_x_1.pack(anchor=tk.W, pady=2)
        self.entry_x_1 = ttk.Entry(panel_controles, font=('Arial', 10), width=20)
        self.entry_x_1.insert(0, "1000") 
        self.entry_x_1.pack(anchor=tk.W, pady=2)
        
        btn_procesar = ttk.Button(panel_controles, text="🔄 Ejecutar Algoritmos Numéricos", command=self.procesar_y_calcular_1)
        btn_procesar.pack(fill=tk.X, pady=10)
        
        self.lbl_res_poly_1 = ttk.Label(panel_controles, text="Y (Lagrange): ---", font=('Arial', 10))
        self.lbl_res_poly_1.pack(anchor=tk.W, pady=2)
        self.lbl_res_spline_1 = ttk.Label(panel_controles, text="Y (Spline): ---", font=('Arial', 10))
        self.lbl_res_spline_1.pack(anchor=tk.W, pady=2)
        
        ttk.Separator(panel_controles, orient='horizontal').pack(fill=tk.X, pady=10)
        
        ttk.Label(panel_controles, text="Ubicación Óptima de Extremos:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=5)
        self.lbl_xmin_1 = ttk.Label(panel_controles, text="Punto Crítico X: ---")
        self.lbl_xmin_1.pack(anchor=tk.W, pady=2)
        self.lbl_ymin_1 = ttk.Label(panel_controles, text="Punto Crítico Y: ---")
        self.lbl_ymin_1.pack(anchor=tk.W, pady=2)
        self.lbl_d2y_1 = ttk.Label(panel_controles, text="Estabilidad (2da Derivada): ---")
        self.lbl_d2y_1.pack(anchor=tk.W, pady=2)
        self.lbl_estado_1 = ttk.Label(panel_controles, text="Estado: ---")
        self.lbl_estado_1.pack(anchor=tk.W, pady=2)
        
        ttk.Separator(panel_controles, orient='horizontal').pack(fill=tk.X, pady=10)
        
        ttk.Label(panel_controles, text="Cambiar Pestaña Gráfica:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=5)
        self.grafica_actual_1 = "A" 
        ttk.Button(panel_controles, text="Vista A: Dispersión y Curva Suave", command=lambda: self.cambiar_grafica_1("A")).pack(fill=tk.X, pady=3)
        ttk.Button(panel_controles, text="Vista B: Análisis de Runge", command=lambda: self.cambiar_grafica_1("B")).pack(fill=tk.X, pady=3)
        ttk.Button(panel_controles, text="Vista C: Monitoreo de Derivadas", command=lambda: self.cambiar_grafica_1("C")).pack(fill=tk.X, pady=3)
        
        self.frame_filtros_b_1 = ttk.LabelFrame(panel_controles, text=" Filtros Vista B (Grados) ", padding=5)
        self.frame_filtros_b_1.pack(fill=tk.X, pady=5)
        self.var_global_1 = tk.BooleanVar(value=True)
        self.var_g5_1 = tk.BooleanVar(value=True)
        self.var_g10_1 = tk.BooleanVar(value=True)
        self.var_g15_1 = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.frame_filtros_b_1, text="Polinomio Global (Max)", variable=self.var_global_1, command=self.actualizar_lienzo_1).pack(anchor=tk.W)
        ttk.Checkbutton(self.frame_filtros_b_1, text="Polinomio Grado 5", variable=self.var_g5_1, command=self.actualizar_lienzo_1).pack(anchor=tk.W)
        ttk.Checkbutton(self.frame_filtros_b_1, text="Polinomio Grado 10", variable=self.var_g10_1, command=self.actualizar_lienzo_1).pack(anchor=tk.W)
        ttk.Checkbutton(self.frame_filtros_b_1, text="Polinomio Grado 15", variable=self.var_g15_1, command=self.actualizar_lienzo_1).pack(anchor=tk.W)
        
        panel_grafico = ttk.LabelFrame(parent, text=" 3. Panel Visualizador Estructural ", padding=5)
        panel_grafico.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.fig_1, self.ax_1 = plt.subplots(figsize=(6, 5))
        self.canvas_1 = FigureCanvasTkAgg(self.fig_1, master=panel_grafico)
        self.canvas_1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.actualizar_nombres_ejes_1()

    def actualizar_nombres_ejes_1(self):
        preset_seleccionado = self.combo_preset.get()
        self.nombre_x_1, self.nombre_y_1 = self.presets_ejes[preset_seleccionado]
        self.lbl_eval_x_1.config(text=f"Evaluar en {self.nombre_x_1.split(' ')[0]}:")

    def extraer_datos_1(self):
        lineas = self.txt_datos_1.get("1.0", tk.END).strip().split('\n')
        x_lista, y_lista = [], []
        for l in lineas:
            if l.strip():
                try:
                    val = l.split(',')
                    x_lista.append(float(val[0].strip()))
                    y_lista.append(float(val[1].strip()))
                except:
                    raise ValueError(f"Error de lectura en la fila: '{l}'")
        if len(x_lista) < 4: raise ValueError("Se requieren al menos 4 puntos en la tabla.")
        return np.array(x_lista, dtype=float), np.array(y_lista, dtype=float)

    def encontrar_extremos_locales_1(self, func_derivada, x_min, x_max, resolucion=1000):
        x_vals = np.linspace(x_min, x_max, resolucion)
        y_vals = func_derivada(x_vals)
        raices = []
        for i in range(len(x_vals)-1):
            if y_vals[i] * y_vals[i+1] < 0:
                raiz_exacta = brentq(func_derivada, x_vals[i], x_vals[i+1])
                raices.append(raiz_exacta)
        return raices

    def calcular_error_loo_1(self, X, Y):
        # Función original intacta
        if len(X) <= 6: return
        np.random.seed(42) 
        idx_test = np.random.choice(range(2, len(X)-2), size=5, replace=False)
        errores = []
        print(f"\n--- VALIDACIÓN LEAVE-ONE-OUT (LOO) ---")
        for i in idx_test:
            X_train = np.delete(X, i)
            Y_train = np.delete(Y, i)
            poly_loo = lagrange(X_train, Y_train)
            Y_pred = poly_loo(X[i])
            error = abs(Y_pred - Y[i]) / abs(Y[i]) if Y[i] != 0 else 0
            errores.append(error)
            print(f"Punto Oculto {X[i]} Hz: Real = {Y[i]} Ohm | Pred = {Y_pred:.4f} Ohm | Error = {error:.2%}")
        print(f"Error Promedio Global: {np.mean(errores):.2%}\n")

    def procesar_y_calcular_1(self):
        try:
            self.X1, self.Y1 = self.extraer_datos_1()
            self.X1_fina = np.linspace(self.X1.min(), self.X1.max(), 1000)
            self.poly_1 = lagrange(self.X1, self.Y1)
            self.spline_1 = CubicSpline(self.X1, self.Y1, bc_type='natural')
            self.d1_spline_1 = self.spline_1.derivative(nu=1)
            self.d2_spline_1 = self.spline_1.derivative(nu=2)
            
            self.calcular_error_loo_1(self.X1, self.Y1)
            
            val_x = float(self.entry_x_1.get())
            self.lbl_res_poly_1.config(text=f"{self.nombre_y_1.split(' ')[0]} (Lagrange): {self.poly_1(val_x):.4f}")
            self.lbl_res_spline_1.config(text=f"{self.nombre_y_1.split(' ')[0]} (Spline): {self.spline_1(val_x):.4f}")
            
            extremos = self.encontrar_extremos_locales_1(self.d1_spline_1, self.X1.min(), self.X1.max())
            self.minimo_encontrado_1 = None
            if extremos:
                for raiz in extremos:
                    if self.d2_spline_1(raiz) > 0:
                        self.minimo_encontrado_1 = raiz
                        break
            
            if self.minimo_encontrado_1 is not None:
                x_min = self.minimo_encontrado_1
                y_min = self.spline_1(x_min)
                self.lbl_xmin_1.config(text=f"Crítico {self.nombre_x_1.split(' ')[0]}: {x_min:.4f}")
                self.lbl_ymin_1.config(text=f"Crítico {self.nombre_y_1.split(' ')[0]}: {y_min:.4f}")
                self.lbl_d2y_1.config(text=f"Derivada Segunda: {self.d2_spline_1(x_min):.4f}")
                self.lbl_estado_1.config(text="Análisis: Mínimo Estable (Signo +)", foreground="green")
            else:
                self.lbl_xmin_1.config(text="Punto Crítico X: No hallado")
                self.lbl_estado_1.config(text="Análisis: Sin extremos locales convexos", foreground="orange")
                
            self.val_x_1 = val_x
            self.actualizar_lienzo_1()
        except ValueError as e:
            messagebox.showerror("Error de Cálculo", str(e))

    def cambiar_grafica_1(self, tipo):
        self.grafica_actual_1 = tipo
        self.actualizar_lienzo_1()

    def actualizar_lienzo_1(self):
        if not hasattr(self, 'X1'): return 
        self.ax_1.clear()
        self.ax_1.grid(True, linestyle='--')
        
        if self.grafica_actual_1 == "A":
            self.ax_1.scatter(self.X1, self.Y1, color='red', label='Puntos Registrados', zorder=5)
            self.ax_1.plot(self.X1_fina, self.spline_1(self.X1_fina), color='black', alpha=0.6, label='Spline de Ajuste')
            self.ax_1.set_title(f'Gráfica Exploratoria: {self.nombre_y_1.split(" ")[0]} vs {self.nombre_x_1.split(" ")[0]}')
        elif self.grafica_actual_1 == "B":
            self.ax_1.scatter(self.X1, self.Y1, color='black', zorder=5, label='Datos')
            n = len(self.X1)
            if self.var_global_1.get(): self.ax_1.plot(self.X1_fina, self.poly_1(self.X1_fina), color='red', label=f'Global (Grado {n-1})')
            if n >= 6 and self.var_g5_1.get():
                p = lagrange(self.X1[np.linspace(0, n-1, 6, dtype=int)], self.Y1[np.linspace(0, n-1, 6, dtype=int)])
                self.ax_1.plot(self.X1_fina, p(self.X1_fina), color='blue', linestyle='--', label='Submodelo (Grado 5)')
            if n >= 11 and self.var_g10_1.get():
                p = lagrange(self.X1[np.linspace(0, n-1, 11, dtype=int)], self.Y1[np.linspace(0, n-1, 11, dtype=int)])
                self.ax_1.plot(self.X1_fina, p(self.X1_fina), color='orange', linestyle='--', label='Submodelo (Grado 10)')
            if n >= 16 and self.var_g15_1.get():
                p = lagrange(self.X1[np.linspace(0, n-1, 16, dtype=int)], self.Y1[np.linspace(0, n-1, 16, dtype=int)])
                self.ax_1.plot(self.X1_fina, p(self.X1_fina), color='green', linestyle='--', label='Submodelo (Grado 15)')
            rango_y = self.Y1.max() - self.Y1.min()
            self.ax_1.set_ylim(self.Y1.min() - rango_y*0.5, self.Y1.max() + rango_y*0.5)
            self.ax_1.set_title('Evidencia Computacional del Fenómeno de Runge')
        elif self.grafica_actual_1 == "C":
            self.ax_1.plot(self.X1_fina, self.d1_spline_1(self.X1_fina), color='purple', label="1ra Derivada Analítica")
            self.ax_1.axhline(0, color='black', linestyle='-')
            if self.minimo_encontrado_1:
                self.ax_1.scatter([self.minimo_encontrado_1], [0], color='green', s=120, zorder=5, label=f"Raíz: {self.minimo_encontrado_1:.2f}")
            self.ax_1.set_title('Comportamiento de la Primera Derivada')

        self.ax_1.axvline(self.val_x_1, color='blue', linestyle=':', label=f'Valor evaluado: {self.val_x_1}')
        self.ax_1.set_xlabel(self.nombre_x_1)
        self.ax_1.set_ylabel(self.nombre_y_1)
        self.ax_1.legend(loc='best')
        self.canvas_1.draw()

    # =================================================================================
    # PESTAÑA 2: NUEVO DASHBOARD PROFESIONAL (EJERCICIO 2)
    # =================================================================================
    def construir_ejercicio_2(self, parent):
        # PANEL IZQUIERDO: Entrada de Datos
        panel_datos = ttk.LabelFrame(parent, text=" 1. Datos Clínicos: f (kHz), V (V), |Z| (Ohm) ", padding=10)
        panel_datos.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10, expand=False)
        
        ttk.Label(panel_datos, text="Formato: f, V, |Z|", font=('Arial', 9, 'italic')).pack(anchor=tk.W, pady=2)
        self.txt_datos_2 = tk.Text(panel_datos, width=35, height=22, font=('Consolas', 10))
        self.txt_datos_2.pack(fill=tk.BOTH, expand=True, pady=5)
        
        datos_2 = (
            "10.0, 0.842, 182.4\n12.5, 0.911, 178.9\n15.0, 0.986, 175.1\n17.5, 1.062, 171.0\n"
            "20.0, 1.143, 166.8\n22.5, 1.227, 162.7\n25.0, 1.314, 158.9\n27.5, 1.401, 155.4\n"
            "30.0, 1.482, 152.0\n32.5, 1.551, 149.0\n35.0, 1.216, 146.1\n37.5, 1.048, 145.2\n"
            "40.0, 0.866, 145.8\n42.5, 0.689, 147.3\n45.0, 0.521, 149.9\n47.5, 0.364, 153.5\n"
            "50.0, 0.223, 158.0\n52.5, 0.103, 163.2\n55.0, 0.012, 168.9\n57.5, -0.041, 174.8\n"
            "60.0, -0.057, 180.5\n62.5, -0.034, 186.2\n65.0, 0.018, 191.5\n67.5, 0.096, 196.2\n"
            "70.0, 0.197, 200.1\n72.5, 0.318, 203.1\n75.0, 0.452, 205.2\n77.5, 0.579, 206.3\n"
            "80.0, 0.700, 206.1\n82.5, 0.809, 204.7\n85.0, 0.611, 198.0\n87.5, 0.688, 194.4\n"
            "90.0, 0.756, 190.9\n92.5, 0.811, 187.8\n95.0, 0.856, 185.1\n97.5, 0.894, 183.0\n"
            "100.0, 0.926, 181.6\n102.5, 0.954, 180.8\n105.0, 0.980, 180.6\n107.5, 1.004, 180.9"
        )
        self.txt_datos_2.insert(tk.END, datos_2)
        
        # Panel Evaluación Dinámica (Debajo del texto)
        frame_dinamico = ttk.Frame(panel_datos)
        frame_dinamico.pack(fill=tk.X, pady=10)
        ttk.Label(frame_dinamico, text="Evaluación Libre f (kHz):", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        self.entry_x_2 = ttk.Entry(frame_dinamico, font=('Arial', 10), width=15)
        self.entry_x_2.insert(0, "41.0")
        self.entry_x_2.pack(anchor=tk.W, pady=2)
        ttk.Button(frame_dinamico, text="▶ Generar Reporte", command=self.procesar_ejercicio_2).pack(fill=tk.X, pady=5)
        
        # PANEL DERECHO: DASHBOARD (Top: Tabla, Bottom: Gráficas)
        panel_derecho = ttk.Frame(parent)
        panel_derecho.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- TABLA DE RESULTADOS EXACTOS (Para el Profesor) ---
        panel_tabla = ttk.LabelFrame(panel_derecho, text=" Reporte Analítico Oficial del Examen ", padding=5)
        panel_tabla.pack(side=tk.TOP, fill=tk.X)
        
        columnas = ("Requerimiento", "Método", "Resultado Calculado")
        self.tree = ttk.Treeview(panel_tabla, columns=columnas, show="headings", height=10)
        self.tree.heading("Requerimiento", text="Parámetro Solicitado")
        self.tree.heading("Método", text="Método Numérico")
        self.tree.heading("Resultado Calculado", text="Valor Obtenido")
        self.tree.column("Requerimiento", width=180)
        self.tree.column("Método", width=150)
        self.tree.column("Resultado Calculado", width=200, anchor=tk.CENTER)
        self.tree.pack(fill=tk.X, expand=True)

        # --- PANEL GRÁFICO ---
        panel_grafico_2 = ttk.LabelFrame(panel_derecho, text=" Visualización de Señales ", padding=5)
        panel_grafico_2.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=10)
        
        frame_botones_graf = ttk.Frame(panel_grafico_2)
        frame_botones_graf.pack(side=tk.TOP, fill=tk.X, pady=5)
        self.grafica_actual_2 = "V"
        ttk.Button(frame_botones_graf, text="[Ver V(f)] Voltaje y Cruces", command=lambda: self.cambiar_grafica_2("V")).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botones_graf, text="[Ver |Z|(f)] Impedancia", command=lambda: self.cambiar_grafica_2("Z")).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botones_graf, text="[Ver dV/df] Sensibilidad", command=lambda: self.cambiar_grafica_2("D")).pack(side=tk.LEFT, padx=5)

        self.fig_2, self.ax_2 = plt.subplots(figsize=(6, 4))
        self.canvas_2 = FigureCanvasTkAgg(self.fig_2, master=panel_grafico_2)
        self.canvas_2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def extraer_datos_2(self):
        lineas = self.txt_datos_2.get("1.0", tk.END).strip().split('\n')
        f_list, v_list, z_list = [], [], []
        for l in lineas:
            if l.strip():
                val = l.split(',')
                f_list.append(float(val[0].strip()))
                v_list.append(float(val[1].strip()))
                z_list.append(float(val[2].strip()))
        return np.array(f_list), np.array(v_list), np.array(z_list)

    def lagrange_3pt(self, f_target, f_data, y_data):
        idx_closest = np.argsort(np.abs(f_data - f_target))[:3]
        f_closest = f_data[idx_closest]
        y_closest = y_data[idx_closest]
        poly = lagrange(f_closest, y_closest)
        return poly(f_target)

    def diff_cent_2(self, y, idx, h): return (y[idx+1] - y[idx-1]) / (2*h)
    def diff_cent_4(self, y, idx, h): return (-y[idx+2] + 8*y[idx+1] - 8*y[idx-1] + y[idx-2]) / (12*h)
    
    def metodo_biseccion(self, f_data, y_data, idx_a, idx_b, tol=1e-5):
        a, b = f_data[idx_a], f_data[idx_b]
        poly_local = lagrange(f_data[idx_a:idx_b+1], y_data[idx_a:idx_b+1])
        if poly_local(a) * poly_local(b) > 0: return None
        while (b - a) / 2.0 > tol:
            c = (a + b) / 2.0
            if poly_local(c) == 0: return c
            elif poly_local(a) * poly_local(c) < 0: b = c
            else: a = c
        return (a + b) / 2.0

    def procesar_ejercicio_2(self):
        try:
            self.F, self.V, self.Z = self.extraer_datos_2()
            self.F_fina = np.linspace(self.F.min(), self.F.max(), 1000)
            
            self.spline_V = CubicSpline(self.F, self.V, bc_type='natural')
            self.spline_Z = CubicSpline(self.F, self.Z, bc_type='natural')
            
            # Limpiar tabla
            for item in self.tree.get_children():
                self.tree.delete(item)
                
            # --- PARTE 1: INTERPOLACIÓN FIJA EN TABLA ---
            for f_val in [41.0, 73.0]:
                self.tree.insert("", "end", values=(f"Voltaje V({f_val} kHz)", "Lagrange (3 ptos)", f"{self.lagrange_3pt(f_val, self.F, self.V):.4f} V"))
                self.tree.insert("", "end", values=(f"Voltaje V({f_val} kHz)", "Spline Cúbico", f"{self.spline_V(f_val):.4f} V"))
                self.tree.insert("", "end", values=(f"Impedancia |Z|({f_val} kHz)", "Lagrange (3 ptos)", f"{self.lagrange_3pt(f_val, self.F, self.Z):.4f} Ohm"))
                self.tree.insert("", "end", values=(f"Impedancia |Z|({f_val} kHz)", "Spline Cúbico", f"{self.spline_Z(f_val):.4f} Ohm"))
            
            # --- PARTE 2: DERIVADAS EN TABLA ---
            h = 2.5
            for f_val in [40.0, 70.0, 100.0]:
                idx = np.where(self.F == f_val)[0][0]
                self.tree.insert("", "end", values=(f"Derivada dV/df en {f_val}", "Dif. Centrada O(h²)", f"{self.diff_cent_2(self.V, idx, h):.4f} V/kHz"))
                self.tree.insert("", "end", values=(f"Derivada dV/df en {f_val}", "Dif. Centrada O(h⁴)", f"{self.diff_cent_4(self.V, idx, h):.4f} V/kHz"))
                self.tree.insert("", "end", values=(f"Derivada dV/df en {f_val}", "Analítica (Spline)", f"{self.spline_V.derivative(1)(f_val):.4f} V/kHz"))
            
            # Progresiva en 10.0
            dp2 = (-3*self.V[0] + 4*self.V[1] - self.V[2]) / (2*h)
            self.tree.insert("", "end", values=("Derivada dV/df en 10.0", "Dif. Progresiva O(h²)", f"{dp2:.4f} V/kHz"))

            # --- PARTE 3: RAÍCES EN TABLA ---
            self.raices_f = []
            cambios = [i for i in range(len(self.V)-1) if self.V[i] * self.V[i+1] < 0]
            for i, idx in enumerate(cambios):
                r_bis = self.metodo_biseccion(self.F, self.V, idx, idx+1)
                r_spl = brentq(self.spline_V, self.F[idx], self.F[idx+1])
                self.raices_f.append(r_spl)
                self.tree.insert("", "end", values=(f"Cruce por Cero #{i+1}", "Bisección", f"{r_bis:.4f} kHz"))
                self.tree.insert("", "end", values=(f"Cruce por Cero #{i+1}", "Raíz Spline", f"{r_spl:.4f} kHz"))
            
            # Datos dinámicos del usuario
            self.val_f_user = float(self.entry_x_2.get())
            self.tree.insert("", "end", values=(f"--- EVALUACIÓN LIBRE ---", "---", "---"))
            self.tree.insert("", "end", values=(f"Voltaje V({self.val_f_user})", "Spline", f"{self.spline_V(self.val_f_user):.4f} V"))
            self.tree.insert("", "end", values=(f"Impedancia |Z|({self.val_f_user})", "Spline", f"{self.spline_Z(self.val_f_user):.4f} Ohm"))

            self.actualizar_lienzo_2()
        except Exception as e:
            messagebox.showerror("Error", f"Fallo al procesar datos: {str(e)}")

    def cambiar_grafica_2(self, tipo):
        self.grafica_actual_2 = tipo
        self.actualizar_lienzo_2()

    def actualizar_lienzo_2(self):
        if not hasattr(self, 'F'): return 
        self.ax_2.clear()
        self.ax_2.grid(True, linestyle='--')
        
        if self.grafica_actual_2 == "V":
            self.ax_2.scatter(self.F, self.V, color='blue', label='Datos Experimentales V(f)')
            self.ax_2.plot(self.F_fina, self.spline_V(self.F_fina), color='black', label='Modelo Spline V(f)')
            self.ax_2.axhline(0, color='red', linestyle='-', linewidth=1.5)
            for r in self.raices_f:
                self.ax_2.scatter([r], [0], color='red', s=100, zorder=5, label=f"Activación Alarma: {r:.2f} kHz")
            self.ax_2.set_title("Comportamiento del Voltaje de Salida")
            self.ax_2.set_ylabel("Voltaje V(f) [V]")
            
        elif self.grafica_actual_2 == "Z":
            self.ax_2.scatter(self.F, self.Z, color='red', label='Datos |Z|(f)')
            self.ax_2.plot(self.F_fina, self.spline_Z(self.F_fina), color='black', label='Spline |Z|(f)')
            self.ax_2.set_title("Magnitud de la Impedancia")
            self.ax_2.set_ylabel("Impedancia |Z| [Ohm]")
            
        elif self.grafica_actual_2 == "D":
            d1 = self.spline_V.derivative(1)
            self.ax_2.plot(self.F_fina, d1(self.F_fina), color='purple', label="Sensibilidad (dV/df)")
            self.ax_2.axhline(0, color='black', linestyle='--')
            self.ax_2.set_title("Derivada de Voltaje (Sensibilidad del Módulo)")
            self.ax_2.set_ylabel("dV/df [V/kHz]")

        self.ax_2.axvline(self.val_f_user, color='green', linestyle=':', label=f'Evaluación Libre: {self.val_f_user}')
        self.ax_2.set_xlabel("Frecuencia f [kHz]")
        
        handles, labels = self.ax_2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax_2.legend(by_label.values(), by_label.keys(), loc='best')
        self.canvas_2.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = AppIntegrada(root)
    # Autocargar gráficos iniciales
    app.procesar_y_calcular_1()
    app.procesar_ejercicio_2()
    root.mainloop()
