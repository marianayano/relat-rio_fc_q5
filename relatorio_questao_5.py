import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Parâmetros do método de Hückel
alpha_C = 0.00    # Energia de sítio para carbono
alpha_N = 0.50    # Energia de sítio para nitrogênio
beta_CC = -1.00   # Integral de salto para C-C
beta_CN = -0.90   # Integral de salto para C-N

# Construir a matriz Hamiltoniana
H = np.zeros((6, 6))

# Preencher diagonais
H[0, 0] = alpha_N  # Nitrogênio (sítio 1)
for i in range(1, 6):
    H[i, i] = alpha_C  # Carbonos (sítios 2-6)

# Preencher acoplamentos de 1º vizinhos
# Ligações C-N
H[0, 1] = beta_CN  # N-C (1-2)
H[1, 0] = beta_CN
H[0, 5] = beta_CN  # N-C (1-6)
H[5, 0] = beta_CN

# Ligações C-C
H[1, 2] = beta_CC  # C-C (2-3)
H[2, 1] = beta_CC
H[2, 3] = beta_CC  # C-C (3-4)
H[3, 2] = beta_CC
H[3, 4] = beta_CC  # C-C (4-5)
H[4, 3] = beta_CC
H[4, 5] = beta_CC  # C-C (5-6)
H[5, 4] = beta_CC

# Diagonalizar a matriz Hamiltoniana
eigvals, eigvecs = eigh(H)

# Ordenar autovalores e autovetores
idx = eigvals.argsort()
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Tarefa 1: Stick plot do espectro
print("=== ESPECTRO DE ENERGIA ===")
print("Autovalores (em unidades de β_CC):")
for i, E in enumerate(eigvals):
    print(f"E_{i+1} = {E:.4f}")

plt.figure(figsize=(10, 6))
for i, E in enumerate(eigvals):
    plt.plot([i+1, i+1], [0, E], 'b-', linewidth=3)
    plt.plot(i+1, E, 'ro', markersize=8)

# Destacar HOMO e LUMO
homo_index = 2  # 3º orbital (índice 2)
lumo_index = 3  # 4º orbital (índice 3)
plt.plot(homo_index+1, eigvals[homo_index], 'go', markersize=10, label='HOMO')
plt.plot(lumo_index+1, eigvals[lumo_index], 'mo', markersize=10, label='LUMO')

plt.axhline(0, color='k', linestyle='-', alpha=0.3)
plt.xlabel('Estado')
plt.ylabel('Energia (unidades de β_CC)')
plt.title('Espectro de Energia da Piridina')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('espectro_piridina.png')
plt.show()

# Tarefa 2: Populações por sítio
print("\n=== POPULAÇÕES POR SÍTIO ===")
# 6 elétrons π → ocupar os 3 orbitais mais baixos
occupied_orbitals = eigvecs[:, :3]  # Primeiros 3 orbitais (ocupados)
q = np.zeros(6)

for i in range(6):  # Para cada sítio
    for mu in range(3):  # Para cada orbital ocupado
        q[i] += 2 * np.abs(eigvecs[i, mu])**2

print("Populações eletrônicas:")
for i in range(6):
    atom_type = "N" if i == 0 else "C"
    print(f"Sítio {i+1} ({atom_type}): q = {q[i]:.4f}")

print(f"Soma total: {np.sum(q):.4f} (esperado: 6.000)")

# Tarefa 3: Ordens de ligação π
print("\n=== ORDENS DE LIGAÇÃO ===")
# Lista de ligações: (i, j, tipo)
bonds = [
    (0, 1, "C-N"),  # N-C (1-2)
    (1, 2, "C-C"),  # C-C (2-3)
    (2, 3, "C-C"),  # C-C (3-4)
    (3, 4, "C-C"),  # C-C (4-5)
    (4, 5, "C-C"),  # C-C (5-6)
    (5, 0, "C-N")   # C-N (6-1)
]

p = {}
for (i, j, bond_type) in bonds:
    p_ij = 0.0
    for mu in range(3):  # Para cada orbital ocupado
        p_ij += 2 * eigvecs[i, mu] * eigvecs[j, mu]
    p[(i, j, bond_type)] = p_ij
    print(f"Ligação {i+1}-{j+1} ({bond_type}): p = {p_ij:.4f}")

# Tarefa 4: Mapas HOMO e LUMO
def plot_orbital(coefficients, title, filename):
    plt.figure(figsize=(8, 8))
    
    # Coordenadas dos átomos em um hexágono
    angles = np.linspace(0, 2*np.pi, 7)[:-1]  # 6 pontos
    x = np.cos(angles)
    y = np.sin(angles)
    
    # Plotar átomos
    for i in range(6):
        atom_type = "N" if i == 0 else "C"
        color = 'red' if i == 0 else 'blue'
        plt.plot(x[i], y[i], 'o', markersize=20, color=color, alpha=0.7)
        plt.text(x[i], y[i], f"{i+1}\n({atom_type})", ha='center', va='center', fontsize=12)
    
    # Plotar ligações
    for (i, j, bond_type) in bonds:
        plt.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=0.5)
    
    # Plotar amplitudes dos orbitais
    max_amplitude = np.max(np.abs(coefficients))
    for i in range(6):
        amplitude = np.abs(coefficients[i])
        phase = np.sign(coefficients[i])
        color = 'red' if phase > 0 else 'blue'
        
        # Bolha proporcional à amplitude
        circle = plt.Circle((x[i], y[i]), amplitude/max_amplitude * 0.3, 
                           color=color, alpha=0.5)
        plt.gca().add_patch(circle)
    
    plt.axis('equal')
    plt.axis('off')
    plt.title(title)
    plt.savefig(filename)
    plt.show()

# Plotar HOMO e LUMO
print("\n=== MAPAS HOMO E LUMO ===")
homo = eigvecs[:, homo_index]
lumo = eigvecs[:, lumo_index]

plot_orbital(homo, 'Orbital HOMO da Piridina', 'homo_piridina.png')
plot_orbital(lumo, 'Orbital LUMO da Piridina', 'lumo_piridina.png')

# Cálculo do gap HOMO-LUMO
E_gap = eigvals[lumo_index] - eigvals[homo_index]
print(f"\nGap HOMO-LUMO: {E_gap:.4f} unidades de β_CC")