import numpy as np

def skew(a):
    """Returns skew symmetric matrix, given a 3-vector"""
    return np.array([
        [    0, -a[2],  a[1]],
        [ a[2],     0, -a[0]],
        [-a[1],  a[0],     0]
        ])   

def rot1(t):
    return np.array([
        [1, 0, 0],
        [0, np.cos(t), -np.sin(t)],
        [0, np.sin(t),  np.cos(t)]
        ])

def rot2(t):
    return np.array([
        [np.cos(t), 0, np.sin(t)],
        [0, 1, 0],
        [-np.sin(t), 0, np.cos(t)]
        ])

def rot3(t):
    return np.array([
        [np.cos(t), -np.sin(t), 0],
        [np.sin(t), np.cos(t), 0],
        [0, 0, 1]
        ])

def rot2D(t):
    return np.array([
        [np.cos(t), -np.sin(t)],
        [np.sin(t),  np.cos(t)]
        ])

def euler2rot(phi, theta, psi):
    return rot3(psi) @ rot2(theta) @ rot1(phi) 

def rot2euler(R):
    """Compute Euler angles from rotation matrix.
    yaw, pitch, roll: 3, 2, 1 rot sequence
    Note frame relationship: e^b = e^v R^{vb}
    """
    psi = np.arctan2(R[1, 0], R[0, 0]) # yaw angle
    theta = np.arcsin(-R[2, 0])        # pitch angle
    phi = np.arctan2(R[2, 1], R[2, 2]) # roll angle
    return (phi, theta, psi)


def conj(q):
    return np.concatenate([[q[0]], -q[1:]])

def prod(p, q):
    """Compute product of two quaternions"""
    p0 = p[0]; p = p[1:4]
    q0 = q[0]; q = q[1:4]
    pq0 = p0*q0 - np.dot(p, q)
    pq = p0*q + p*q0 + np.cross(p,q)
    return np.concatenate([[pq0], pq])

def euler2quat(phi, theta, psi):
    psi2 = psi/2
    theta2 = theta/2
    phi2 = phi/2
    return np.array([
        np.sin(phi2)*np.sin(psi2)*np.sin(theta2) + np.cos(phi2)*np.cos(psi2)*np.cos(theta2), 
        np.sin(phi2)*np.cos(psi2)*np.cos(theta2) - np.sin(psi2)*np.sin(theta2)*np.cos(phi2), 
        np.sin(phi2)*np.sin(psi2)*np.cos(theta2) + np.sin(theta2)*np.cos(phi2)*np.cos(psi2), 
        -np.sin(phi2)*np.sin(theta2)*np.cos(psi2) + np.sin(psi2)*np.cos(phi2)*np.cos(theta2)
        ])

def quat2rot(q):
    """Compute rotation matrix from quaternion.
    quaternion must be provided in form [q0, q]
    """    
    q = q.flatten()
    q0 = q[0]
    q = q[1:]
    return (q0**2 - np.dot(q, q))*np.eye(3) + 2*np.outer(q,q) + 2*q0*skew(q)

def quat2euler(q):
    R = quat2rot(q)
    return rot2euler(R)

if __name__ == "__main__":    
    # Tests
    
    # Check quat_prod
    p = np.array([3, 1, -2, 1])
    q = np.array([2, -1, 2, 3])
    pq = np.array([8, -9, -2, 11])
    print('pq_hand - pq = ', np.linalg.norm(prod(p,q)-pq))
    
    # Check conversions rot <-> quat
    phi = 2*np.pi * np.random.random_sample()
    theta = 2*np.pi * np.random.random_sample()
    psi = 2*np.pi * np.random.random_sample()
    R = euler2rot(phi=phi, theta=theta, psi=psi)
    q = euler2quat(phi=phi, theta=theta, psi=psi)
    R_ = quat2rot(q)
    print('R - R_ = ',np.linalg.norm(R-R_))
 
    # Check defs w and qdot
    w = np.random.rand(4); w[0]=0
    qdot = 0.5 * prod(q, w)
    qc = conj(q)
    w_ = 2 * prod(qc, qdot)
    print('w - w_ = ', np.linalg.norm(w-w_))

    phi_, theta_, psi_ = quat2euler(q)
    print("phi - phi_ = ", phi - phi_)
    print("theta - theta_ = ", theta - theta_)
    print("psi - psi_ = ", psi - psi_)


    