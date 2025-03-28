# Importar las librerías necesarias
from pycaret.classification import *
import pandas as pd

# Cargar la data
ruta_archivo = r"C:\Users\DIEGOFUENTES\Desktop\SEMANA2\DATA\diabetes.csv"
df = pd.read_csv(ruta_archivo)

# Inicializar PyCaret con la variable objetivo 'Outcome'
clf_setup = setup(data=df, target='Outcome', session_id=123, log_experiment=False)

# Comparar modelos y seleccionar el mejor
mejor_modelo = compare_models()

# Crear un reporte de desempeño de todos los modelos
resultados_modelos = pull()
print("Comparación de Modelos:\n", resultados_modelos)

# Entrenar el mejor modelo
modelo_final = create_model(mejor_modelo)

# Hacer predicciones con el mejor modelo
predicciones = predict_model(modelo_final)
print("Ejemplo de Predicciones:\n", predicciones.head())

# Ajuste de hiperparámetros (tuning)
modelo_tuneado = tune_model(modelo_final, optimize='Accuracy')

# Evaluar el modelo ajustado
evaluate_model(modelo_tuneado)

# Guardar el modelo entrenado
save_model(modelo_tuneado, "mejor_modelo_pycaret")

# 1. Inicializo Git en mi carpeta de trabajo si aún no está inicializado
# En la terminal de VS Code ejecuto:
# git init

# 2. Agrego todos los archivos al área de preparación
# Esto asegura que Git rastree todos los cambios:
# git add .

# 3. Hago un commit con un mensaje descriptivo
# Esto guarda los cambios localmente en Git:
# git commit -m "Subiendo script de clasificación con PyCaret"

# 4. Conecto mi proyecto con un repositorio en GitHub

# 5. Subo mi código a GitHub
# Aseguro que la rama principal se llame 'main' y empujo los cambios:
# git branch -M main
# git push -u origin main
