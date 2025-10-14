import rclpy, math
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

def median_bin_5x5(image, n):
    """
    Perform 5x5 median binning on a mono image.

    Args:
        image (numpy.ndarray): Input 2D mono image.

    Returns:
        numpy.ndarray: Downsampled image after 5x5 median binning.
    """
    # Ensure the image dimensions are divisible by 5
    h, w = image.shape
    h_new, w_new = h // n, w // n
    image = image[:h_new * n, :w_new * n]  # Crop to make divisible by 5

    # Reshape into 5x5 blocks
    reshaped = image.reshape(h_new, n, w_new, n)

    # Compute the median for each 5x5 block
    #binned = np.median(reshaped, axis=(1, 3)).astype(np.uint16)
    binned = np.median(reshaped, axis=(1, 3))

    return binned

def disparity_to_3d(disparity, f, B, cx, cy, n):
    """
    Converts a disparity image to 3D points using NumPy.

    Args:
        disparity (numpy.ndarray): Disparity image (2D array).
        f (float): Focal length of the camera.
        B (float): Baseline (distance between the stereo cameras).
        cx (float): Principal point x-coordinate.
        cy (float): Principal point y-coordinate.

    Returns:
        numpy.ndarray: Nx3 array of 3D points.
    """
    f = f / n
    cx = cx /n
    cy = cy / n

    # Get the image dimensions
    h, w = disparity.shape

    # Create a grid of pixel coordinates
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    # Avoid division by zero by masking invalid disparity values
    # ingnore any point 3m away
    valid_mask = disparity >= 24

    # Compute depth (Z)
    Z = np.zeros_like(disparity, dtype=np.float32)
    Z[valid_mask] = (f * B) / (disparity[valid_mask]*0.125/n)

    # Compute X and Y
    X = (x_coords - cx) * Z / f
    Y = (y_coords - cy) * Z / f

    # Stack X, Y, Z into an Nx3 array of 3D points
    points_3d = np.stack((Z[valid_mask], -X[valid_mask], -Y[valid_mask]), axis=-1)

    return points_3d

def disp_callback(img_msg):
    #print(type(img_msg.data))
    n=5
    binned = median_bin_5x5(np.frombuffer(img_msg.data, dtype=np.uint16).reshape(img_msg.height, img_msg.width), n)
    #print(binned.shape, binned.dtype)
    p_3d = disparity_to_3d(binned, 470.051, 0.0750492, 314.96, 229.359, n)
    header = Header()
    header.frame_id = "body"
    header.stamp = img_msg.header.stamp
    pc_pub.publish(point_cloud2.create_cloud_xyz32(header, p_3d))

#    img = Image()
#    img.header = header
#    img.height = 480 // n
#    img.width = 640 // n
#    img.is_bigendian = 0
#    img.encoding = "mono8"
#    img.step = img.width
#    img.data = binned.astype(np.uint8).ravel()
#    img_pub.publish(img)

rclpy.init()
node = rclpy.create_node('disp_pc')

best_effort_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
pc_pub = node.create_publisher(PointCloud2, "obstacles", best_effort_qos)
disp_sub = node.create_subscription(Image, "disparity", disp_callback, qos_profile=best_effort_qos)
#img_pub = node.create_publisher(Image, "binned", best_effort_qos)

while rclpy.ok():
    try:
        rclpy.spin_once(node)
    except KeyboardInterrupt:
        break
rclpy.try_shutdown()
print("bye")
