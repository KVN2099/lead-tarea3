#!/usr/bin/env python3
"""
estadisticas_mpi.py
Programa para calcular estadísticas globales usando operaciones colectivas MPI.

Autor: [Tu nombre]
Fecha: 2025-07-27
"""

from mpi4py import MPI
import numpy as np
import sys

def main():
    # Inicializar MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Verificar argumentos de línea de comandos
    if len(sys.argv) != 2:
        if rank == 0:
            print("Uso: mpirun -np <num_procesos> python estadisticas_mpi.py <tamaño_arreglo>")
        return
    
    try:
        N = int(sys.argv[1])
    except ValueError:
        if rank == 0:
            print("Error: El tamaño del arreglo debe ser un número entero")
        return
    
    # Verificar que N sea divisible entre el número de procesos
    if N % size != 0:
        if rank == 0:
            print(f"Error: El tamaño del arreglo ({N}) debe ser divisible entre el número de procesos ({size})")
        return
    
    # Calcular el tamaño del subarreglo para cada proceso
    subarray_size = N // size
    
    # Proceso raíz inicializa el arreglo
    if rank == 0:
        # Generar números aleatorios entre 0 y 100
        np.random.seed(42)  # Para resultados reproducibles
        data = np.random.uniform(0, 100, N).astype(np.float64)
        print(f"Arreglo de {N} elementos generado en proceso raíz")
        print(f"Distribuyendo entre {size} procesos ({subarray_size} elementos por proceso)")
    else:
        data = None
    
    # Broadcast del tamaño del subarreglo (aunque ya lo calculamos, es parte del ejercicio)
    subarray_size = comm.bcast(subarray_size, root=0)
    
    # Preparar buffer para recibir datos
    local_data = np.empty(subarray_size, dtype=np.float64)
    
    # Scatter: distribuir partes del arreglo entre procesos
    comm.Scatter(data, local_data, root=0)
    
    # Cada proceso calcula estadísticas locales
    local_min = np.min(local_data)
    local_max = np.max(local_data)
    local_sum = np.sum(local_data)
    local_count = len(local_data)
    
    print(f"Proceso {rank}: min={local_min:.2f}, max={local_max:.2f}, promedio={local_sum/local_count:.2f}")
    
    # Reduce: calcular estadísticas globales
    global_min = comm.reduce(local_min, op=MPI.MIN, root=0)
    global_max = comm.reduce(local_max, op=MPI.MAX, root=0)
    global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
    total_count = comm.reduce(local_count, op=MPI.SUM, root=0)
    
    # El proceso raíz imprime los resultados finales
    if rank == 0:
        global_avg = global_sum / total_count
        print("\n" + "="*50)
        print("ESTADÍSTICAS GLOBALES:")
        print(f"Mínimo global: {global_min:.2f}")
        print(f"Máximo global: {global_max:.2f}")
        print(f"Promedio global: {global_avg:.2f}")
        print(f"Total de elementos procesados: {total_count}")
        print("="*50)
    
    # Opcional: Gather para reconstruir el arreglo completo
    if rank == 0:
        reconstructed = np.empty(N, dtype=np.float64)
    else:
        reconstructed = None
    
    comm.Gather(local_data, reconstructed, root=0)
    
    if rank == 0:
        # Verificar que la reconstrucción es correcta
        if np.allclose(data, reconstructed):
            print("✓ Verificación: El arreglo se reconstruyó correctamente con Gather")
        else:
            print("✗ Error: El arreglo reconstruido no coincide con el original")

if __name__ == "__main__":
    main()
