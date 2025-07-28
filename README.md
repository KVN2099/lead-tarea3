# Tarea 3: Medición de Latencia y Comunicación Colectiva MPI

Este repositorio contiene la implementación de programas MPI en Python para medir latencia de comunicaciones y realizar operaciones colectivas.

## Estudiante
[Kevin Romero]  
Cartago, Costa Rica  
Fecha: 2025-07-27

## Descripción

El proyecto consta de tres programas principales:

1. **estadisticas_mpi.py**: Calcula estadísticas globales usando operaciones colectivas MPI
2. **latencia_mpi.py**: Mide la latencia de comunicaciones punto a punto
3. **graficar_latencia.py**: [Opcional] Genera gráficos de latencia vs tamaño de mensaje

## Requisitos del Sistema

- Python 3.6+
- mpi4py
- numpy
- matplotlib (solo para el programa opcional)
- Una implementación de MPI (OpenMPI, MPICH, etc.)

### Instalación de dependencias

```pip3 install mpi4py numpy matplotlib```


## Uso

### Parte A: Estadísticas Globales

Calcula el mínimo, máximo y promedio de un arreglo distribuido entre procesos.

```mpirun -np 4 python estadisticas_mpi.py 1000000```


**Parámetros:**
- `-np`: Número de procesos MPI
- `1000000`: Tamaño del arreglo (debe ser divisible entre el número de procesos)

### Parte B: Medición de Latencia

Mide la latencia de comunicación entre dos procesos.

```mpirun -np 2 python latencia_mpi.py iteraciones```


**Parámetros:**
- `-np 2`: Exactamente 2 procesos (requerido)
- `iteraciones`: Número de iteraciones para el promedio (opcional, por defecto 10000)


### Opcional: Generación de Gráficos

Genera gráficos detallados de latencia vs tamaño de mensaje.

```mpirun -np 2 python graficar_latencia.py```


Esto generará:
- `latencia_vs_tamaño.png`: Gráfico con escalas log-log y semi-log
- `datos_latencia.csv`: Datos en formato CSV para análisis posterior

## Características Técnicas

### Operaciones MPI Utilizadas

**Parte A:**
- `MPI_Bcast`: Difusión del tamaño del subarreglo
- `MPI_Scatter`: Distribución de datos entre procesos
- `MPI_Reduce`: Agregación de estadísticas (MIN, MAX, SUM)
- `MPI_Gather`: Reconstrucción del arreglo completo

**Parte B:**
- `MPI_Send`/`MPI_Recv`: Comunicación punto a punto
- `MPI_Wtime`: Medición precisa de tiempo
- `MPI_Barrier`: Sincronización de procesos

### Buenas Prácticas Implementadas

1. **Manejo de errores**: Validación de argumentos y configuración MPI
2. **Documentación**: Comentarios explicativos y docstrings
3. **Modularidad**: Funciones separadas para cada funcionalidad
4. **Reproducibilidad**: Semilla fija para números aleatorios
5. **Escalabilidad**: Ajuste automático de iteraciones según tamaño de mensaje
6. **Verificación**: Comprobación de resultados en Parte A

## Análisis de Resultados

### Parte A: Operaciones Colectivas

Las operaciones colectivas son más eficientes que las comunicaciones punto a punto equivalentes porque:
- Están optimizadas para la topología de red específica
- Minimizan el número de saltos de red
- Pueden usar algoritmos de árbol o pipeline

### Parte B: Latencia de Comunicaciones

Los resultados típicos muestran:
- **Latencia base**: 5-50 microsegundos (overhead del protocolo MPI)
- **Transición**: Alrededor de 1KB-10KB donde el ancho de banda domina
- **Mensajes grandes**: Tiempo proporcional al tamaño del mensaje

## Posibles Variaciones en los Resultados

1. **Hardware**: CPU, memoria, red de interconexión
2. **Software**: Implementación de MPI, versión de Python
3. **Sistema**: Carga del sistema, otros procesos ejecutándose
4. **Red**: Latencia y ancho de banda de la red
5. **Configuración**: Número de procesos, distribución en nodos

## Referencias

- [mpi4py Documentation](https://mpi4py.readthedocs.io/)
- [MPI Standard](https://www.mpi-forum.org/)
- [OpenMPI Documentation](https://www.open-mpi.org/doc/)