import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Importaciones clave para los métodos numéricos del curso
from scipy.interpolate import lagrange, CubicSpline
from scipy.optimize import brentq
import warnings

# Ignorar advertencias genéricas (específicamente RankWarning en polinomios de alto grado mal condicionados)
# Esto limpia la terminal para presentar un software más profesional.
warnings.simplefilter('ignore')

class InterpoladorUniversalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Software Multipropósito de Modelado e Interpolación Numérica")
        self.root.geometry("1250x760")
        
        style = ttk.Style()
        style.theme_use('clam')
        
        # Diccionario para adaptar la interfaz a diferentes contextos físicos/ingenieriles
        self.presets_ejes = {
            "General (X vs Y)": ("Variable Independiente (X)", "Variable Dependiente (Y)"),
            "Biomédica (Frecuencia vs Impedancia)": ("Frecuencia f (Hz)", "Magnitud de Impedancia |Z| (Ohm)"),
            "Cinemática (Tiempo vs Posición)": ("Tiempo t (s)", "Posición x (m)"),
            "Termodinámica (Temperatura vs Presión)": ("Temperatura T (°C)", "Presión P (kPa)")
        }
        
        # --- PANEL IZQUIERDO: ENTRADA DE DATOS ---
        # Interfaz de entrada para los puntos (nodos) del experimento
        panel_datos = ttk.LabelFrame(root, text=" 1. Base de Datos del Experimento ", padding=10)
        panel_datos.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10, expand=False)
        
        ttk.Label(panel_datos, text="Formato: X, Y\n(Una medición por cada línea):", 
                  font=('Arial', 9, 'italic')).pack(anchor=tk.W, pady=2)
        
        self.txt_datos = tk.Text(panel_datos, width=32, height=25, font=('Consolas', 10))
        self.txt_datos.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Datos precargados del problema biomédico
        datos_iniciales = (
            "100, 152.3\n120, 149.1\n145, 146.8\n170, 144.9\n200, 142.0\n"
            "235, 139.5\n270, 137.9\n310, 136.1\n355, 134.8\n405, 133.6\n"
            "460, 132.7\n520, 131.9\n585, 131.4\n655, 131.1\n730, 130.9\n"
            "810, 131.0\n895, 131.3\n985, 131.9\n1080, 132.7\n1180, 133.8\n"
            "1290, 135.2\n1410, 136.9\n1540, 138.9\n1680, 141.1\n1830, 143.5\n"
            "1990, 146.1\n2160, 149.0\n2340, 152.2\n2530, 155.6\n2730, 159.2"
        )
        self.txt_datos.insert(tk.END, datos_iniciales)
        
        # --- PANEL CENTRAL: MOTOR DE CÁLCULO ---
        # Controles para seleccionar métodos numéricos y mostrar resultados
        panel_controles = ttk.LabelFrame(root, text=" 2. Configuración y Cálculos ", padding=10)
        panel_controles.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=10)
        
        ttk.Label(panel_controles, text="Contexto / Tipo de Estudio:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=2)
        self.combo_preset = ttk.Combobox(panel_controles, values=list(self.presets_ejes.keys()), state="readonly", width=35)
        self.combo_preset.set("Biomédica (Frecuencia vs Impedancia)") 
        self.combo_preset.pack(anchor=tk.W, pady=5)
        self.combo_preset.bind("<<ComboboxSelected>>", lambda e: self.actualizar_nombres_ejes())
        
        ttk.Separator(panel_controles, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Punto a interpolar
        self.lbl_eval_x = ttk.Label(panel_controles, text="Evaluar en X:", font=('Arial', 10, 'bold'))
        self.lbl_eval_x.pack(anchor=tk.W, pady=2)
        self.entry_x = ttk.Entry(panel_controles, font=('Arial', 10), width=20)
        self.entry_x.insert(0, "1000") 
        self.entry_x.pack(anchor=tk.W, pady=2)
        
        btn_procesar = ttk.Button(panel_controles, text="🔄 Ejecutar Algoritmos Numéricos", command=self.procesar_y_calcular)
        btn_procesar.pack(fill=tk.X, pady=10)
        
        # Etiquetas de resultados de interpolación
        self.lbl_res_poly = ttk.Label(panel_controles, text="Y (Lagrange): ---", font=('Arial', 10))
        self.lbl_res_poly.pack(anchor=tk.W, pady=2)
        self.lbl_res_spline = ttk.Label(panel_controles, text="Y (Spline): ---", font=('Arial', 10))
        self.lbl_res_spline.pack(anchor=tk.W, pady=2)
        
        ttk.Separator(panel_controles, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Panel de visualización de cálculo de derivadas (Mínimos locales)
        ttk.Label(panel_controles, text="Ubicación Óptima de Extremos:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=5)
        self.lbl_xmin = ttk.Label(panel_controles, text="Punto Crítico X: ---")
        self.lbl_xmin.pack(anchor=tk.W, pady=2)
        self.lbl_ymin = ttk.Label(panel_controles, text="Punto Crítico Y: ---")
        self.lbl_ymin.pack(anchor=tk.W, pady=2)
        self.lbl_d2y = ttk.Label(panel_controles, text="Estabilidad (2da Derivada): ---")
        self.lbl_d2y.pack(anchor=tk.W, pady=2)
        self.lbl_estado = ttk.Label(panel_controles, text="Estado: ---")
        self.lbl_estado.pack(anchor=tk.W, pady=2)
        
        ttk.Separator(panel_controles, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Selector de Vistas Gráficas
        ttk.Label(panel_controles, text="Cambiar Pestaña Gráfica:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=5)
        self.grafica_actual = "A" 
        ttk.Button(panel_controles, text="Vista A: Dispersión y Curva Suave", command=lambda: self.cambiar_grafica("A")).pack(fill=tk.X, pady=3)
        ttk.Button(panel_controles, text="Vista B: Análisis de Runge", command=lambda: self.cambiar_grafica("B")).pack(fill=tk.X, pady=3)
        ttk.Button(panel_controles, text="Vista C: Monitoreo de Derivadas", command=lambda: self.cambiar_grafica("C")).pack(fill=tk.X, pady=3)
        
        # Filtros para activar/desactivar submodelos de Lagrange y evidenciar el Fenómeno de Runge
        self.frame_filtros_b = ttk.LabelFrame(panel_controles, text=" Filtros Vista B (Grados) ", padding=5)
        self.frame_filtros_b.pack(fill=tk.X, pady=5)
        
        self.var_global = tk.BooleanVar(value=True)
        self.var_g5 = tk.BooleanVar(value=True)
        self.var_g10 = tk.BooleanVar(value=True)
        self.var_g15 = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(self.frame_filtros_b, text="Polinomio Global (Max)", variable=self.var_global, command=self.actualizar_lienzo).pack(anchor=tk.W)
        ttk.Checkbutton(self.frame_filtros_b, text="Polinomio Grado 5", variable=self.var_g5, command=self.actualizar_lienzo).pack(anchor=tk.W)
        ttk.Checkbutton(self.frame_filtros_b, text="Polinomio Grado 10", variable=self.var_g10, command=self.actualizar_lienzo).pack(anchor=tk.W)
        ttk.Checkbutton(self.frame_filtros_b, text="Polinomio Grado 15", variable=self.var_g15, command=self.actualizar_lienzo).pack(anchor=tk.W)
        
        # --- PANEL DERECHO: VISTA GRÁFICA ---
        panel_grafico = ttk.LabelFrame(root, text=" 3. Panel Visualizador Estructural ", padding=5)
        panel_grafico.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=panel_grafico)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.actualizar_nombres_ejes()

    def actualizar_nombres_ejes(self):
        # Actualiza las etiquetas de la interfaz según el contexto elegido
        preset_seleccionado = self.combo_preset.get()
        self.nombre_x, self.nombre_y = self.presets_ejes[preset_seleccionado]
        self.lbl_eval_x.config(text=f"Evaluar en {self.nombre_x.split(' ')[0]}:")
        self.procesar_y_calcular()

    def extraer_datos(self):
        # Parsea el bloque de texto y lo convierte en arreglos de NumPy para su procesamiento matemático
        lineas = self.txt_datos.get("1.0", tk.END).strip().split('\n')
        x_lista, y_lista = [], []
        for l in lineas:
            if l.strip():
                try:
                    val = l.split(',')
                    x_lista.append(float(val[0].strip()))
                    y_lista.append(float(val[1].strip()))
                except:
                    raise ValueError(f"Error de lectura en la fila: '{l}'. Use el formato numérico: número, número")
        if len(x_lista) < 4:
            raise ValueError("Se requieren al menos 4 puntos en la tabla.")
        return np.array(x_lista, dtype=float), np.array(y_lista, dtype=float)

    def encontrar_extremos_locales(self, func_derivada, x_min, x_max, resolucion=1000):
        # METODO NUMÉRICO PARA BÚSQUEDA DE RAÍCES:
        # Busca los puntos donde la primera derivada cruza el eje cero (cambio de signo)
        x_vals = np.linspace(x_min, x_max, resolucion)
        y_vals = func_derivada(x_vals)
        raices = []
        for i in range(len(x_vals)-1):
            if y_vals[i] * y_vals[i+1] < 0:
                # Aplica el método de Brent (brentq) que es una combinación robusta
                # del método de Bisección, Secante e Interpolación Cuadrática Inversa.
                raiz_exacta = brentq(func_derivada, x_vals[i], x_vals[i+1])
                raices.append(raiz_exacta)
        return raices

    def calcular_error_loo(self, X, Y):
        # VALIDACIÓN LEAVE-ONE-OUT (LOO):
        # Demuestra matemáticamente la falta de predictibilidad de Lagrange.
        # Oculta un punto aleatorio, construye el polinomio con el resto de datos
        # y compara la predicción vs el valor real.
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
            print(f"Punto Oculto {X[i]} Hz: Real = {Y[i]} Ohm | Predicción Lagrange = {Y_pred:.4f} Ohm | Error Relativo = {error:.2%}")
        print(f"Error Promedio Global: {np.mean(errores):.2%}\n")

    def procesar_y_calcular(self):
        try:
            # 1. Extracción y preparación de la malla continua para graficar
            self.X, self.Y = self.extraer_datos()
            self.X_fina = np.linspace(self.X.min(), self.X.max(), 1000)
            
            # 2. CÁLCULO DE MODELOS:
            # Construye el Polinomio Interpolante de Lagrange global
            self.poly = lagrange(self.X, self.Y)
            # Construye el Spline Cúbico Natural (S''=0 en los extremos)
            self.spline = CubicSpline(self.X, self.Y, bc_type='natural')
            
            # 3. CÁLCULO DE DERIVADAS ANALÍTICAS DEL SPLINE
            self.d1_spline = self.spline.derivative(nu=1) # Primera derivada (Pendiente)
            self.d2_spline = self.spline.derivative(nu=2) # Segunda derivada (Curvatura)
            
            # 4. Análisis de error e interpolación solicitada por el usuario
            self.calcular_error_loo(self.X, self.Y)
            val_x = float(self.entry_x.get())
            self.lbl_res_poly.config(text=f"{self.nombre_y.split(' ')[0]} (Lagrange): {self.poly(val_x):.4f}")
            self.lbl_res_spline.config(text=f"{self.nombre_y.split(' ')[0]} (Spline): {self.spline(val_x):.4f}")
            
            # 5. ANÁLISIS DE PUNTOS CRÍTICOS:
            extremos = self.encontrar_extremos_locales(self.d1_spline, self.X.min(), self.X.max())
            self.minimo_encontrado = None
            
            if extremos:
                # Evalúa las raíces con el criterio de la Segunda Derivada
                for raiz in extremos:
                    if self.d2_spline(raiz) > 0: # Si S''(x) > 0, es un mínimo local convexo
                        self.minimo_encontrado = raiz
                        break
            
            # Muestra los resultados físicos/matemáticos del extremo en la interfaz
            if self.minimo_encontrado is not None:
                x_min = self.minimo_encontrado
                y_min = self.spline(x_min)
                d2y_min = self.d2_spline(x_min)
                
                self.lbl_xmin.config(text=f"Crítico {self.nombre_x.split(' ')[0]}: {x_min:.4f}")
                self.lbl_ymin.config(text=f"Crítico {self.nombre_y.split(' ')[0]}: {y_min:.4f}")
                self.lbl_d2y.config(text=f"Derivada Segunda: {d2y_min:.4f}")
                self.lbl_estado.config(text="Análisis: Mínimo Estable (Signo +)", foreground="green")
            else:
                self.lbl_xmin.config(text="Punto Crítico X: No hallado")
                self.lbl_ymin.config(text="Punto Crítico Y: ---")
                self.lbl_d2y.config(text="Derivada Segunda: ---")
                self.lbl_estado.config(text="Análisis: Sin extremos locales convexos", foreground="orange")
                
            self.val_x = val_x
            self.actualizar_lienzo()
            
        except ValueError as e:
            messagebox.showerror("Error de Cálculo", str(e))

    def cambiar_grafica(self, tipo):
        # Actualiza el tipo de vista seleccionado por el usuario en el panel
        self.grafica_actual = tipo
        self.actualizar_lienzo()

    def actualizar_lienzo(self):
        if not hasattr(self, 'X'):
            return 
        
        self.ax.clear()
        self.ax.grid(True, linestyle='--')
        
        # VISTA A: Ajuste de Spline vs Datos Reales
        if self.grafica_actual == "A":
            self.ax.scatter(self.X, self.Y, color='red', label='Puntos Registrados', zorder=5)
            self.ax.plot(self.X_fina, self.spline(self.X_fina), color='black', alpha=0.6, label='Spline de Ajuste')
            self.ax.set_title(f'Gráfica Exploratoria: {self.nombre_y.split(" ")[0]} vs {self.nombre_x.split(" ")[0]}')
            
        # VISTA B: Demostración del Fenómeno de Runge (Inestabilidad Polinómica)
        elif self.grafica_actual == "B":
            self.ax.scatter(self.X, self.Y, color='black', zorder=5, label='Datos')
            n = len(self.X)
            
            # Grafica el polinomio de interpolación global
            if self.var_global.get():
                self.ax.plot(self.X_fina, self.poly(self.X_fina), color='red', label=f'Polinomio Global (Grado {n-1})')
                
            # Subsistemas (submuestreo) para demostrar cómo empeora la oscilación a mayor grado
            if n >= 6 and self.var_g5.get():
                idx = np.linspace(0, n-1, 6, dtype=int)
                p = lagrange(self.X[idx], self.Y[idx])
                self.ax.plot(self.X_fina, p(self.X_fina), color='blue', linestyle='--', label='Submodelo (Grado 5)')
                
            if n >= 11 and self.var_g10.get():
                idx = np.linspace(0, n-1, 11, dtype=int)
                p = lagrange(self.X[idx], self.Y[idx])
                self.ax.plot(self.X_fina, p(self.X_fina), color='orange', linestyle='--', label='Submodelo (Grado 10)')
                
            if n >= 16 and self.var_g15.get():
                idx = np.linspace(0, n-1, 16, dtype=int)
                p = lagrange(self.X[idx], self.Y[idx])
                self.ax.plot(self.X_fina, p(self.X_fina), color='green', linestyle='--', label='Submodelo (Grado 15)')
            
            # Limitador del eje Y para que los colapsos infinitos de Runge no rompan la gráfica
            rango_y = self.Y.max() - self.Y.min()
            self.ax.set_ylim(self.Y.min() - rango_y*0.5, self.Y.max() + rango_y*0.5)
            self.ax.set_title('Evidencia Computacional del Fenómeno de Runge')

        # VISTA C: Gráfica de la función primera derivada
        elif self.grafica_actual == "C":
            self.ax.plot(self.X_fina, self.d1_spline(self.X_fina), color='purple', label="1ra Derivada Numérica (Analítica)")
            self.ax.axhline(0, color='black', linestyle='-', linewidth=1) # Eje horizontal 0
            if self.minimo_encontrado:
                # Dibuja un punto verde donde la pendiente de la curva cruza por cero
                self.ax.scatter([self.minimo_encontrado], [0], color='green', s=120, zorder=5, label=f"Cruce por Cero (Raíz: {self.minimo_encontrado:.2f})")
            self.ax.set_title('Comportamiento Continuo de la Primera Derivada')

        # Línea vertical para ubicar gráficamente el punto evaluado por el usuario
        self.ax.axvline(self.val_x, color='blue', linestyle=':', label=f'Valor evaluado: {self.val_x}')
        self.ax.set_xlabel(self.nombre_x)
        self.ax.set_ylabel(self.nombre_y)
        self.ax.legend(loc='best')
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = InterpoladorUniversalApp(root)
    root.mainloop()