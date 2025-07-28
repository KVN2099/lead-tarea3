#!/usr/bin/env python3
"""
latencia_mpi.py
Programa para medir la latencia de comunicaciones punto a punto en MPI.

Autor: [Tu nombre]
Fecha: 2025-07-27
"""

from mpi4py import MPI
import numpy as np
import sys
import time

def measure_latency(comm, rank, message_size, iterations=10000):
    """
    Mide la latencia para mensajes de un tamaño específico.
    
    Args:
        comm: Comunicador MPI
        rank: Rango del proceso
        message_size: Tamaño del mensaje en bytes
        iterations: Número de iteraciones para el promedio
    
    Returns:
        latencia_promedio: Latencia promedio por mensaje (ida y vuelta) en microsegundos
    """
    
    if rank == 0:
        # Proceso 0: envía y recibe mensajes
        message = np.ones(message_size, dtype=np.uint8)  # Mensaje de bytes
        received = np.empty(message_size, dtype=np.uint8)
        
        # Sincronizar antes de empezar la medición
        comm.Barrier()
        
        start_time = MPI.Wtime()
        
        for i in range(iterations):
            # Enviar mensaje al proceso 1
            comm.Send(message, dest=1, tag=0)
            # Recibir mensaje de vuelta del proceso 1
            comm.Recv(received, source=1, tag=1)
        
        end_time = MPI.Wtime()
        
        total_time = end_time - start_time
        latency_per_message = (total_time / iterations) * 1e6  # Convertir a microsegundos
        
        return latency_per_message
    
    elif rank == 1:
        # Proceso 1: recibe y retorna mensajes
        message = np.empty(message_size, dtype=np.uint8)
        
        # Sincronizar antes de empezar la medición
        comm.Barrier()
        
        for i in range(iterations):
            # Recibir mensaje del proceso 0
            comm.Recv(message, source=0, tag=0)
            # Enviar mensaje de vuelta al proceso 0
            comm.Send(message, dest=0, tag=1)
        
        return 0  # El proceso 1 no calcula latencia

def main():
    # Inicializar MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Verificar que solo se usen 2 procesos
    if size != 2:
        if rank == 0:
            print("Error: Este programa requiere exactamente 2 procesos")
            print("Uso: mpirun -np 2 python latencia_mpi.py [iteraciones]")
        return
    
    # Número de iteraciones (por defecto 10000)
    iterations = 10000
    if len(sys.argv) > 1:
        try:
            iterations = int(sys.argv[1])
        except ValueError:
            if rank == 0:
                print("Advertencia: Número de iteraciones inválido, usando 10000")
    
    if rank == 0:
        print(f"Midiendo latencia con {iterations} iteraciones...")
        print("="*60)
    
    # Medición básica: mensaje de 1 byte
    latency = measure_latency(comm, rank, 1, iterations)
    
    if rank == 0:
        print(f"Mensaje de 1 byte transmitido {iterations} veces.")
        print(f"Latencia promedio por mensaje (ida y vuelta): {latency:.2f} microsegundos")
        print(f"Latencia estimada unidireccional: {latency/2:.2f} microsegundos")
        print()
    
    # Parte opcional: medición para diferentes tamaños
    if rank == 0:
        print("MEDICIONES PARA DIFERENTES TAMAÑOS DE MENSAJE:")
        print("-" * 60)
        print(f"{'Tamaño':<15} {'Latencia (μs)':<15} {'Unidireccional (μs)':<20}")
        print("-" * 60)
    
    # Tamaños de mensaje para análisis
    message_sizes = [
        (1, "1 B"),
        (1024, "1 KB"),
        (1024*1024, "1 MB")
    ]
    
    results = []
    
    for size_bytes, size_label in message_sizes:
        # Usar menos iteraciones para mensajes grandes para evitar tiempos excesivos
        current_iterations = max(100, iterations // (size_bytes // 1024 + 1))
        
        latency = measure_latency(comm, rank, size_bytes, current_iterations)
        
        if rank == 0:
            unidirectional = latency / 2
            results.append((size_bytes, size_label, latency, unidirectional))
            print(f"{size_label:<15} {latency:<15.2f} {unidirectional:<20.2f}")
    
    if rank == 0:
        print("-" * 60)
        print("\nANÁLISIS:")
        print("- Para mensajes pequeños, la latencia es dominada por el overhead de comunicación")
        print("- Para mensajes grandes, el tiempo de transferencia domina sobre la latencia")
        print("- La transición ocurre típicamente alrededor de 1KB-10KB dependiendo del hardware")
        
        # Análisis simple de los resultados
        if len(results) >= 2:
            ratio_1kb_1b = results[1][2] / results[0][2]
            print(f"- Ratio latencia 1KB/1B: {ratio_1kb_1b:.1f}x")
            
            if len(results) >= 3:
                ratio_1mb_1kb = results[2][2] / results[1][2]
                print(f"- Ratio latencia 1MB/1KB: {ratio_1mb_1kb:.1f}x")

if __name__ == "__main__":
    main()
