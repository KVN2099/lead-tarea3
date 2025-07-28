#!/usr/bin/env python3
"""
graficar_latencia.py
Programa opcional para generar gráficos de latencia vs tamaño de mensaje.

Autor: [Tu nombre]
Fecha: 2025-07-27
"""

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import sys

def measure_latency_range(comm, rank, max_iterations=1000):
    """Mide latencia para un rango de tamaños de mensaje."""
    
    # Tamaños de mensaje desde 1 byte hasta 10 MB
    sizes = [2**i for i in range(0, 24)]  # 1B, 2B, 4B, ..., 8MB, 16MB
    sizes = [s for s in sizes if s <= 10*1024*1024]  # Limitar a 10MB
    
    latencies = []
    
    for size in sizes:
        # Ajustar iteraciones según el tamaño del mensaje
        iterations = max(10, max_iterations // max(1, size // 1024))
        
        if rank == 0:
            message = np.ones(size, dtype=np.uint8)
            received = np.empty(size, dtype=np.uint8)
            
            comm.Barrier()
            start_time = MPI.Wtime()
            
            for _ in range(iterations):
                comm.Send(message, dest=1, tag=0)
                comm.Recv(received, source=1, tag=1)
            
            end_time = MPI.Wtime()
            latency = (end_time - start_time) / iterations * 1e6  # microsegundos
            latencies.append(latency)
            
        elif rank == 1:
            message = np.empty(size, dtype=np.uint8)
            
            comm.Barrier()
            
            for _ in range(iterations):
                comm.Recv(message, source=0, tag=0)
                comm.Send(message, dest=0, tag=1)
    
    return sizes, latencies

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size != 2:
        if rank == 0:
            print("Error: Este programa requiere exactamente 2 procesos")
        return
    
    if rank == 0:
        print("Generando datos para gráfico de latencia...")
    
    sizes, latencies = measure_latency_range(comm, rank)
    
    if rank == 0:
        # Crear gráfico
        plt.figure(figsize=(12, 8))
        
        # Gráfico en escala log-log
        plt.subplot(2, 1, 1)
        plt.loglog(sizes, latencies, 'b-o', markersize=4)
        plt.xlabel('Tamaño del mensaje (bytes)')
        plt.ylabel('Latencia (microsegundos)')
        plt.title('Latencia vs Tamaño del Mensaje (Escala Log-Log)')
        plt.grid(True, alpha=0.3)
        
        # Gráfico en escala lineal
        plt.subplot(2, 1, 2)
        plt.semilogx(sizes, latencies, 'r-o', markersize=4)
        plt.xlabel('Tamaño del mensaje (bytes)')
        plt.ylabel('Latencia (microsegundos)')
        plt.title('Latencia vs Tamaño del Mensaje (Escala Semi-Log)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('latencia_vs_tamaño.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Gráfico guardado como 'latencia_vs_tamaño.png'")
        
        # Guardar datos en archivo CSV
        with open('datos_latencia.csv', 'w') as f:
            f.write('Tamaño_bytes,Latencia_microsegundos\n')
            for s, l in zip(sizes, latencies):
                f.write(f'{s},{l:.4f}\n')
        
        print("Datos guardados en 'datos_latencia.csv'")

if __name__ == "__main__":
    main()
