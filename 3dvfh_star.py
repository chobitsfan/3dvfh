import rclpy, math
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, PointCloud
from geometry_msgs.msg import TwistStamped, PointStamped
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

def vfh_star_3d_pointcloud_target_direction(point_cloud, target_direction, prv_yaw, prv_pitch, bin_size=10, safety_distance=1.0, alpha=0.5, prv_weight=0.1, openspace_threshold = 5.0):
    """
    Implements a simplified 3D Vector Field Histogram* (VFH*) for UAV obstacle avoidance,
    using 3D point cloud and a target direction vector as input, returning a normalized 3D vector.

    Args:
        point_cloud (numpy.ndarray): Nx3 array of 3D points (x, y, z).
        target_direction (numpy.ndarray): 3D vector representing the target direction.
        bin_size (int): Size of each histogram bin in degrees.
        max_range (float): Maximum range of the sensor.
        safety_distance (float): Minimum safe distance from obstacles.
        alpha (float): Weighting factor for target direction influence (0 to 1).
        valley_threshold (float): Threshold for identifying valleys in the histogram.

    Returns:
        numpy.ndarray: Normalized 3D vector representing the best direction.
    """

    yaw_counts = 360 // bin_size
    pitch_counts = 180 // bin_size

    # Normalize the target direction.
    normalized_direction = target_direction / np.linalg.norm(target_direction)

    # Convert normalized direction to yaw and pitch.
    pitch_target = math.asin(normalized_direction[2])
    yaw_target = math.atan2(normalized_direction[1], normalized_direction[0])

    if prv_yaw is None:
        prv_yaw = yaw_target
        prv_pitch = pitch_target

    # 1. Histogram Creation
    histogram = np.zeros((pitch_counts, yaw_counts), dtype=np.float32) # pitch x yaw

    # Compute depth (distance) for all points
    depth = np.sqrt(np.sum(point_cloud**2, axis=1))

    # Convert to spherical coordinates
    xy2 = np.sum(point_cloud[:, :2]**2, axis=1)  # x^2 + y^2
    yaw = np.arctan2(point_cloud[:, 1], point_cloud[:, 0])  # atan2(y, x)
    pitch = np.arctan2(point_cloud[:, 2], np.sqrt(xy2))  # atan2(z, sqrt(x^2 + y^2))

    # Calculate magnitude (influence) of the obstacles
    magnitude = (safety_distance / depth)**2

    # Bin the obstacles into the histogram
    yaw_bin = ((np.degrees(yaw) + 180) // bin_size).astype(int) % yaw_counts
    pitch_bin = ((np.degrees(pitch) + 90) // bin_size).astype(int) % pitch_counts

    # Accumulate magnitudes into the histogram
    np.add.at(histogram, (pitch_bin, yaw_bin), magnitude)

    # Define the yaw range (in degrees)
    yaw_min = -30  # Minimum yaw angle
    yaw_max = 30   # Maximum yaw angle

    # Convert yaw range to bins
    yaw_min_bin = int((yaw_min + 180) // bin_size) % yaw_counts
    yaw_max_bin = int((yaw_max + 180) // bin_size) % yaw_counts

    # Define the pitch range (in degrees)
    pitch_min = -30  # Minimum pitch angle
    pitch_max = 30   # Maximum pitch angle

    # Convert pitch range to bins
    pitch_min_bin = int((pitch_min + 90) // bin_size) % pitch_counts
    pitch_max_bin = int((pitch_max + 90) // bin_size) % pitch_counts

#    to_inflate = []
#    for yaw_bin in range(yaw_min_bin, yaw_max_bin):
#        for pitch_bin in range(pitch_min_bin, pitch_max_bin):  # Restrict pitch_bin to the specified range
#            big = np.max(histogram[pitch_bin-1:pitch_bin+2, yaw_bin-1:yaw_bin+2])
#            if big > 1.0 and histogram[pitch_bin, yaw_bin] / big < 0.1:
#                to_inflate.append((pitch_bin, yaw_bin, big))
#    for ff in to_inflate:
        #print("from", histogram[ff[0], ff[1]], "to", ff[2])
#        histogram[ff[0], ff[1]] = ff[2]
    #print(len(to_inflate))

    # 2. Polar Histogram Reduction (open space)
    openspace_mask = np.zeros_like(histogram, dtype=bool)
    for yaw_bin in range(yaw_min_bin, yaw_max_bin):
        for pitch_bin in range(pitch_min_bin, pitch_max_bin):
            if np.max(histogram[pitch_bin-1:pitch_bin+2, yaw_bin-1:yaw_bin+2]) < openspace_threshold:
                openspace_mask[pitch_bin, yaw_bin] = True

    # Define the yaw range (in degrees)
    yaw_min = -25  # Minimum yaw angle
    yaw_max = 25   # Maximum yaw angle

    # Convert yaw range to bins
    yaw_min_bin = int((yaw_min + 180) // bin_size) % yaw_counts
    yaw_max_bin = int((yaw_max + 180) // bin_size) % yaw_counts

    # Define the pitch range (in degrees)
    pitch_min = -20  # Minimum pitch angle
    pitch_max = 20   # Maximum pitch angle

    # Convert pitch range to bins
    pitch_min_bin = int((pitch_min + 90) // bin_size) % pitch_counts
    pitch_max_bin = int((pitch_max + 90) // bin_size) % pitch_counts

    # 3. Target Direction Selection (VFH* Modification)
    yaw_target_bin = int((math.degrees(yaw_target) + 180) // bin_size) % yaw_counts
    pitch_target_bin = int((math.degrees(pitch_target) + 90) // bin_size) % pitch_counts

    best_yaw_bin, best_pitch_bin = yaw_target_bin, pitch_target_bin
    min_cost = float('inf')

    prv_yaw_bin = int((math.degrees(prv_yaw) + 180) // bin_size) % yaw_counts
    prv_pitch_bin = int((math.degrees(prv_pitch) + 90) // bin_size) % pitch_counts
    for yaw_bin in range(yaw_min_bin, yaw_max_bin):
        for pitch_bin in range(pitch_min_bin, pitch_max_bin):  # Restrict pitch_bin to the specified range
            if openspace_mask[pitch_bin, yaw_bin]:
                # VFH* cost function: obstacle density + weighted distance from target, prioritize valleys.
                cost = histogram[pitch_bin, yaw_bin] + alpha * math.sqrt((yaw_bin - yaw_target_bin)**2 + (pitch_bin - pitch_target_bin)**2)

                # Favor previous yaw by adding a penalty for deviation from prv_yaw
                cost = cost + prv_weight * math.sqrt((yaw_bin - prv_yaw_bin)**2 + (pitch_bin - prv_pitch_bin)**2)

                if cost < min_cost:
                    min_cost = cost
                    best_yaw_bin, best_pitch_bin = yaw_bin, pitch_bin
    #print(min_cost)

    hist = histogram[10:25, 25:45][::-1, ::-1]*5
    img = Image()
    img.header.stamp = node.get_clock().now().to_msg()
    img.height = hist.shape[0]
    img.width = hist.shape[1]
    img.is_bigendian = 0
    img.encoding = "mono8"
    img.step = img.width
    img.data = hist.astype(np.uint8).ravel()
    hist_pub.publish(img)

    if math.isinf(min_cost):
        return None, None

    # Convert back to radians.
    best_yaw = math.radians(best_yaw_bin * bin_size - 180 + bin_size / 2)
    best_pitch = math.radians(best_pitch_bin * bin_size - 90 + bin_size / 2)

    return best_yaw, best_pitch

def median_bin(image, n):
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
    cx = cx / n
    cy = cy / n

    # Get the image dimensions
    h, w = disparity.shape

    # Create a grid of pixel coordinates
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    # Avoid division by zero by masking invalid disparity values
    # ingnore any point 3m away
    valid_mask = disparity > 90

    # Compute depth (Z)
    # 3bit subpixel disparity = 0.125
    Z = np.zeros_like(disparity, dtype=np.float32)
    Z[valid_mask] = (f * B) / (disparity[valid_mask] * 0.125 / n)

    # Compute X and Y
    X = (x_coords - cx) * Z / f
    Y = (y_coords - cy) * Z / f

    # Stack X, Y, Z into an Nx3 array of 3D points
    points_3d = np.stack((Z[valid_mask], -X[valid_mask], -Y[valid_mask]), axis=-1)

    return points_3d

def obs_callback(msg):
    #print(type(msg.points))
    #for p in msg.points:
    #    print(p.x, p.y, p.z)
    global latest_obs
    latest_obs = msg.points

def disp_callback(img_msg):
    global latest_obs
    n=5
    binned = median_bin(np.frombuffer(img_msg.data, dtype=np.uint16).reshape(img_msg.height, img_msg.width), n)
    latest_obs = disparity_to_3d(binned, 470.051, 0.0750492, 314.96, 229.359, n)
    header = Header()
    header.frame_id = "body"
    header.stamp = node.get_clock().now().to_msg()
    pc_pub.publish(point_cloud2.create_cloud_xyz32(header, latest_obs))

rclpy.init()
node = rclpy.create_node('obs_avd')

# Define a point in the "map" frame
point_in_map = PointStamped()
point_in_map.header.frame_id = "map"
point_in_map.header.stamp = node.get_clock().now().to_msg()
point_in_map.point.x = 5.0
point_in_map.point.y = 0.0
point_in_map.point.z = 1.0

latest_obs = None

# Create a TF2 buffer and listener
#my_tf_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=100)
tf_buffer = Buffer()
tf_listener = TransformListener(tf_buffer, node, spin_thread=False)

best_effort_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.VOLATILE)
#obs_sub = node.create_subscription(PointCloud, "obstacles", obs_callback, 1)
pc_pub = node.create_publisher(PointCloud2, "obstacles", best_effort_qos)
disp_sub = node.create_subscription(Image, "disparity", disp_callback, qos_profile=best_effort_qos)
avd_pub = node.create_publisher(TwistStamped, "avoid_direction", 1)
hist_pub = node.create_publisher(Image, "histogram", best_effort_qos)

prv_yaw = None
prv_pitch = None

while rclpy.ok():
    try:
        rclpy.spin_once(node)
        if latest_obs is not None:
            try:
                # Lookup the transform from "map" to "body"
                transform = tf_buffer.lookup_transform(
                    "body",  # Target frame
                    "map",   # Source frame
                    rclpy.time.Time(),  # Use the latest available transform
                    timeout=rclpy.duration.Duration(seconds=0.0)
                )
            except Exception as e:
                #print(e)
                pass
            else:
                # Transform the point
                point_in_body = do_transform_point(point_in_map, transform)
                best_yaw, best_pitch = vfh_star_3d_pointcloud_target_direction(latest_obs, np.array([point_in_body.point.x, point_in_body.point.y, point_in_body.point.z]), prv_yaw, prv_pitch, safety_distance=1.0, alpha=0.2, prv_weight=0.2, bin_size=5)
                prv_yaw = best_yaw
                prv_pitch = best_pitch

                if best_yaw is None:
                    avd_vel = (0, 0, 0)
                else:
                    # Convert spherical coordinates to a 3D vector.
                    x = math.cos(best_pitch) * math.cos(best_yaw)
                    y = math.cos(best_pitch) * math.sin(best_yaw)
                    z = math.sin(best_pitch)

                    # Normalize the vector.
                    v = np.array([x, y, z])
                    avd_dir = v / np.linalg.norm(v)

                    avd_vel = avd_dir * 0.3
                m = TwistStamped()
                m.header.frame_id = "body"
                m.header.stamp = node.get_clock().now().to_msg()
                m.twist.linear.x = avd_vel[0]
                m.twist.linear.y = avd_vel[1]
                m.twist.linear.z = avd_vel[2]
                avd_pub.publish(m)

                latest_obs = None
    except KeyboardInterrupt:
        break
rclpy.try_shutdown()
print("bye")


