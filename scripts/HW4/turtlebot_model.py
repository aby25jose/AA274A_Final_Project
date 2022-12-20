import numpy as np

EPSILON_OMEGA = 1e-3

def compute_Gx(xvec, u, dt):
    """
    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
    Outputs:
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
    """
    ########## Code starts here ##########
    # TODO: Compute Gx
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    Gx = np.zeros((3, 3))
    if np.abs(u[1]) < EPSILON_OMEGA:
        Gx[0, 0] = 1
        Gx[0, 2] = -u[0]*dt*np.sin(xvec[2])
        Gx[1, 1] = 1
        Gx[1, 2] = u[0]*dt*np.cos(xvec[2])
        Gx[2, 2] = 1
    else:
        Gx[0, 0] = 1
        Gx[0, 2] = u[0] * (np.cos(xvec[2] + u[1]*dt) - np.cos(xvec[2])) / u[1]
        Gx[1, 1] = 1
        Gx[1, 2] = u[0] * (np.sin(xvec[2] + u[1]*dt) - np.sin(xvec[2])) / u[1]
        Gx[2, 2] = 1
        
    ########## Code ends here ##########
    return Gx
    

def compute_Gu(xvec, u, dt):
    """
    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
    Outputs:
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute Gu
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    Gu = np.zeros((3, 2))
    if np.abs(u[1]) < EPSILON_OMEGA:
        Gu[0, 0] = np.cos(xvec[2])*dt
        Gu[0, 1] = -(u[0]*np.sin(xvec[2])*dt**2)/2
        Gu[1, 0] = np.sin(xvec[2])*dt
        Gu[1, 1] = (u[0] * np.cos(xvec[2]) * dt ** 2) / 2
        Gu[2, 1] = dt
    else:
        Gu[0, 0] = (np.sin(xvec[2] + u[1]*dt) - np.sin(xvec[2]))/u[1]
        Gu[0, 1] = (u[1]*u[0]*dt*np.cos(xvec[2] + u[1]*dt) - u[0]*np.sin(xvec[2] + u[1]*dt) + u[0]*np.sin(xvec[2]))/(u[1]**2)
        Gu[1, 0] = (-np.cos(xvec[2] + u[1]*dt) + np.cos(xvec[2]))/u[1]
        Gu[1, 1] = (u[1]*u[0]*dt*np.sin(xvec[2] + u[1]*dt) + u[0]*np.cos(xvec[2] + u[1]*dt) - u[0]*np.cos(xvec[2]))/(u[1]**2)
        Gu[2, 1] = dt
        
    ########## Code ends here ##########
    return Gu


def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    g = np.zeros(3)
    if np.abs(u[1]) < EPSILON_OMEGA:
        g[0] = xvec[0] + u[0]*np.cos(xvec[2])*dt
        g[1] = xvec[1] + u[0] * np.sin(xvec[2]) * dt
        g[2] = xvec[2] + u[1]*dt

    else:
        g[0] = xvec[0] + u[0] * (np.sin(xvec[2] + u[1]*dt) - np.sin(xvec[2])) /u[1]
        g[1] = xvec[1] + u[0] * (-np.cos(xvec[2] + u[1]*dt) + np.cos(xvec[2])) /u[1]
        g[2] = xvec[2] + u[1] * dt

    Gx = compute_Gx(xvec, u, dt)
    Gu = compute_Gu(xvec, u, dt)

    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)
    xcam, ycam, thcam = tf_base_to_camera
    xb, yb, thb = x
    h = np.zeros(2)
    h[0] = alpha - thcam - thb
    h[1] = r - xb*np.cos(alpha) - xcam*np.cos(thb)*np.cos(alpha) + ycam*np.sin(thb)*np.cos(alpha) - yb*np.sin(alpha) - xcam*np.sin(alpha)*np.sin(thb) - ycam*np.cos(thb)*np.sin(alpha)
    Hx = np.zeros((2, 3))
    Hx[0, 0] = 0
    Hx[0, 1] = 0
    Hx[0, 2] = -1
    Hx[1, 0] = -np.cos(alpha)
    Hx[1, 1] = -np.sin(alpha)
    Hx[1, 2] = xcam*np.sin(thb)*np.cos(alpha) + ycam*np.cos(thb)*np.cos(alpha) - xcam*np.sin(alpha)*np.cos(thb) + ycam*np.sin(thb)*np.sin(alpha)

    ########## Code ends here ##########


    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
