import numpy as np
import matplotlib.pyplot as plt

# PBC (Periodic Boundary Conditions)
def get_next(i, N):
    # Returns the next index in a periodic boundary condition array
    return 0 if i == N - 1 else i + 1


def angle_diff(tn, t0):
    # Computes the difference between two angles, ensuring it stays within the range [-π, π]
    dt = tn - t0
    if dt > np.pi:
        dt -= 2 * np.pi
    elif dt < -np.pi:
        dt += 2 * np.pi  
    return dt


def find_defects(phi, phi0=0.8 * (2 * np.pi)):
    """
    Identifies defects in a polar field.
    Parameters:
        phi: 2D array of angles (phase field)
        phi0: Threshold angle for defect detection
    Returns:
        qx, qy: Coordinates of detected defects
        qi: Charges of defects
        defect_phase: Orientation phase of detected defects
    """
    sx, sy = phi.shape  # Get the size of the phase field
    qx, qy, qi, defect_phase = [], [], [], []  # Initialize lists to store defect data
    latt_0 = [-3 * np.pi / 4, -np.pi / 4, np.pi / 4, 3 * np.pi / 4]  # Lattice orientation offsets
    
    for j in range(sy):  # Loop over the y-axis
        jnext = get_next(j, sy)  # Get the next y-index under PBC
        for i in range(sx):  # Loop over the x-axis
            inext = get_next(i, sx)  # Get the next x-index under PBC
            t1 = phi[i, j]  # Current angle
            t2 = phi[inext, j]  # Right neighbor
            t3 = phi[inext, jnext]  # Bottom-right neighbor
            t4 = phi[i, jnext]  # Bottom neighbor
            
            # Sum of angle differences along the lattice loop
            dphi = angle_diff(t2, t1) + angle_diff(t3, t2) + angle_diff(t4, t3) + angle_diff(t1, t4)
            
            # Check if the summed angle difference exceeds the threshold
            if np.abs(dphi) > phi0:
                c = int(np.round(dphi / (2 * np.pi)))  # Quantize the defect charge
                qx.append(i)  # Store x-coordinate
                qy.append(j)  # Store y-coordinate
                qi.append(c)  # Store charge
                
                # Compute the phase (orientation) of the defect
                phase = np.sum(np.exp(1.0j * (np.array([t1, t2, t3, t4]) - c * np.array(latt_0))))
                defect_phase.append(np.mod(np.angle(phase), 2 * np.pi))  # Phase in [0, 2π]
    
    return np.array(qx), np.array(qy), np.array(qi), defect_phase


# Load x and y components of the polar field
nx = np.loadtxt("nx_100000.txt")
ny = np.loadtxt("ny_100000.txt")
phi = np.arctan2(ny, nx)  # Compute the angle (phase) field from the components

# Find defects in the polar field
qx, qy, qi, defect_phase = find_defects(phi, phi0=0.8 * (2 * np.pi))

# Plotting
plt.figure(figsize=(10, 10))
# Plot the polar field using quiver, with arrows representing the direction
plt.quiver(np.arange(phi.shape[0]).T, np.arange(phi.shape[1]).T, np.cos(phi).T, np.sin(phi).T, scale=50, alpha=0.6)

# Plot positive defects
positive_defects_indices = np.where(qi == 1)  # Identify indices of positive defects
scatter = plt.scatter(qx[positive_defects_indices], qy[positive_defects_indices], 
                      c=np.array(defect_phase)[positive_defects_indices], 
                      s=100, label="Positive Defects (+1)", marker='o', vmin=0, vmax=2*np.pi,cmap='hsv')

# Plot negative defects
negative_defects_indices = np.where(qi == -1)  # Identify indices of negative defects
plt.scatter(qx[negative_defects_indices], qy[negative_defects_indices], s=100, edgecolor='k', label="Negative Defects (-1)", marker='^')

# Add colorbar for defect orientation phase
plt.colorbar(scatter)

# Add titles and labels
plt.title("Defects in Polar Field")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.tight_layout()
plt.show()

# Print defect phases and charges for positive defects
print(np.array(defect_phase)[positive_defects_indices])
print("Defect charges:", qi)
print("Defect orientations (phase):", defect_phase)
print("Sum of defect charges:", qi)

# Save defect charges and phases to text files
np.savetxt("charge.txt", qi)
np.savetxt("phase.txt", defect_phase)
