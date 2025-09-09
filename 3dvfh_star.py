import rclpy, math
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, PointCloud
from geometry_msgs.msg import TwistStamped, PointStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point, do_transform_vector3
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from scipy import ndimage

latest_obs = None
disp_ts = None
tgt_p_body = None

class VFH3D:
    def __init__(self, bin_size):
        self.bin_size = bin_size
        yaw_counts = 360 // bin_size
        pitch_counts = 180 // bin_size
        self.histogram = np.zeros((pitch_counts, yaw_counts), dtype=np.float32) # pitch x yaw
        self.occupied_memory = np.zeros_like(self.histogram, dtype=np.int8)
        self.costs = np.zeros_like(self.histogram, dtype=np.float32)
        self.prv_yaw = None
        self.prv_pitch = None

        # Define the yaw range (in degrees)
        yaw_min = -30  # Minimum yaw angle
        yaw_max = 30   # Maximum yaw angle

        # Convert yaw range to bins
        self.yaw_min_bin = int((yaw_min + 180) // self.bin_size) % yaw_counts
        self.yaw_max_bin = int((yaw_max + 180) // self.bin_size) % yaw_counts

        # Define the pitch range (in degrees)
        pitch_min = -25  # Minimum pitch angle
        pitch_max = 25   # Maximum pitch angle

        # Convert pitch range to bins
        self.pitch_min_bin = int((pitch_min + 90) // self.bin_size) % pitch_counts
        self.pitch_max_bin = int((pitch_max + 90) // self.bin_size) % pitch_counts

    def reset(self):
        self.histogram[:] = 0
        self.occupied_memory[:] = 0
        self.prv_yaw = None
        self.prv_pitch = None

    def target_direction(self, point_cloud, yaw_target, pitch_target, safety_distance=1.0, alpha=1.1, occupied_threshold=10.0):
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

        yaw_counts = 360 // self.bin_size
        pitch_counts = 180 // self.bin_size

        if self.prv_yaw is None:
            self.prv_yaw = yaw_target
            self.prv_pitch = pitch_target

        # 1. Histogram Creation
        self.histogram[:] = 0

        # Compute depth (distance) for all points
        depth = np.sqrt(np.sum(point_cloud**2, axis=1))

        # Convert to spherical coordinates
        xy2 = np.sum(point_cloud[:, :2]**2, axis=1)  # x^2 + y^2
        yaw = np.arctan2(point_cloud[:, 1], point_cloud[:, 0])  # atan2(y, x)
        pitch = np.arctan2(point_cloud[:, 2], np.sqrt(xy2))  # atan2(z, sqrt(x^2 + y^2))

        # Calculate magnitude (influence) of the obstacles
        magnitude = (safety_distance / depth)**2

        # Bin the obstacles into the histogram
        yaw_bin = ((np.degrees(yaw) + 180) // self.bin_size).astype(int) % yaw_counts
        pitch_bin = ((np.degrees(pitch) + 90) // self.bin_size).astype(int) % pitch_counts

        # Accumulate magnitudes into the histogram
        np.add.at(self.histogram, (pitch_bin, yaw_bin), magnitude)

        #print(np.max(self.histogram[self.pitch_min_bin:self.pitch_max_bin+1,self.yaw_min_bin:self.yaw_max_bin+1]))
        evade = False
        #if (np.max(self.histogram[self.pitch_min_bin:self.pitch_max_bin+1,self.yaw_min_bin:self.yaw_max_bin+1]) > 150):
        #    evade = True
        # 2. Polar Histogram Reduction (occupied)
#        to_inflates = []
#        for yaw_bin in range(yaw_min_bin, yaw_max_bin):
#            for pitch_bin in range(pitch_min_bin, pitch_max_bin):
#                if occupied[pitch_bin-1, yaw_bin] or occupied[pitch_bin+1, yaw_bin] or occupied[pitch_bin, yaw_bin-1] or occupied[pitch_bin, yaw_bin+1]:
#                    to_inflates.append((pitch_bin, yaw_bin))
#        for to_inflate in to_inflates:
#            occupied[to_inflate[0], to_inflate[1]] = True
        #occupied = self.histogram > occupied_threshold
        #occupied = ndimage.maximum_filter(occupied, size=3)
        self.histogram = ndimage.maximum_filter(self.histogram, size=3)

        self.occupied_memory = np.where(self.histogram > occupied_threshold, 10, self.occupied_memory - 1)
        self.occupied_memory = np.maximum(self.occupied_memory, 0)

        # 3. Target Direction Selection (VFH* Modification)
        yaw_target_bin = int((math.degrees(yaw_target) + 180) // self.bin_size) % yaw_counts
        pitch_target_bin = int((math.degrees(pitch_target) + 90) // self.bin_size) % pitch_counts

        best_yaw_bin, best_pitch_bin = yaw_target_bin, pitch_target_bin
        min_cost = float('inf')
        self.costs[:] = 255

        prv_yaw_bin = int((math.degrees(self.prv_yaw) + 180) // self.bin_size) % yaw_counts
        prv_pitch_bin = int((math.degrees(self.prv_pitch) + 90) // self.bin_size) % pitch_counts
        for yaw_bin in range(self.yaw_min_bin, self.yaw_max_bin+1):
            for pitch_bin in range(self.pitch_min_bin, self.pitch_max_bin+1):  # Restrict pitch_bin to the specified range
                if self.occupied_memory[pitch_bin, yaw_bin] < 1:
                    cost = self.histogram[pitch_bin, yaw_bin] + alpha * math.sqrt((yaw_bin - yaw_target_bin)**2 + (pitch_bin - pitch_target_bin)**2) + math.sqrt((yaw_bin - prv_yaw_bin)**2 + (pitch_bin - prv_pitch_bin)**2)
                    if self.histogram[pitch_bin-1, yaw_bin] < 5 and self.histogram[pitch_bin+1, yaw_bin] < 5 or self.histogram[pitch_bin, yaw_bin-1] < 5 and self.histogram[pitch_bin, yaw_bin+1] < 5:
                        cost = cost * 0.5

                    self.costs[pitch_bin, yaw_bin] = cost

                    if cost < min_cost:
                        min_cost = cost
                        best_yaw_bin, best_pitch_bin = yaw_bin, pitch_bin
        #print(min_cost)

        if math.isinf(min_cost):
            return None, None, None

        # Convert back to radians.
        best_yaw = math.radians(best_yaw_bin * self.bin_size - 180 + self.bin_size / 2)
        best_pitch = math.radians(best_pitch_bin * self.bin_size - 90 + self.bin_size / 2)

        self.prv_yaw = best_yaw
        self.prv_pitch = best_pitch

        return best_yaw, best_pitch, evade

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
    valid_mask = disparity >= 24

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

def disp_callback(img_msg):
    global latest_obs
    global disp_ts
    disp_ts = img_msg.header.stamp
    n=5
    binned = median_bin(np.frombuffer(img_msg.data, dtype=np.uint16).reshape(img_msg.height, img_msg.width), n)
    latest_obs = disparity_to_3d(binned, 470.051, 0.0750492, 314.96, 229.359, n)

def tgt_point_callback(p_msg):
    global tgt_p_body
    tgt_p_body = p_msg

def main():
    global latest_obs
    global tgt_p_body
    rclpy.init()
    node = rclpy.create_node('obs_avd')

    tgt_p_map = None

    # Create a TF2 buffer and listener
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node, spin_thread=False)

    best_effort_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.VOLATILE)
    pc_pub = node.create_publisher(PointCloud2, "obstacles", best_effort_qos)
    avd_pub = node.create_publisher(TwistStamped, "avoid_direction", best_effort_qos)
    hist_pub = node.create_publisher(Image, "histogram", best_effort_qos)
    cost_pub = node.create_publisher(Image, "cost", best_effort_qos)
    disp_sub = node.create_subscription(Image, "disparity", disp_callback, qos_profile=best_effort_qos)
    tgt_point_sub = node.create_subscription(PointStamped, "target_point", tgt_point_callback, qos_profile=best_effort_qos)

    vfh3d = VFH3D(5)

    while rclpy.ok():
        try:
            rclpy.spin_once(node)
            if latest_obs is not None:
                header = Header()
                header.frame_id = "body"
                header.stamp = disp_ts
                pc_pub.publish(point_cloud2.create_cloud_xyz32(header, latest_obs))
                if tgt_p_body is not None:
                    if tgt_p_body.point.x == 0 and tgt_p_body.point.y == 0 and tgt_p_body.point.z == 0:
                        tgt_p_map = None
                        tgt_p_body = None
                        vfh3d.reset()
                    else:
                        try:
                            transform = tf_buffer.lookup_transform(
                                "map",  # Target frame
                                "body",   # Source frame
                                rclpy.time.Time(),  # Use the latest available transform
                                timeout=rclpy.duration.Duration(seconds=0.0)
                            )
                        except Exception as e:
                            pass
                        else:
                            tgt_p_map = do_transform_point(tgt_p_body, transform)
                            tgt_p_body = None
                if tgt_p_map is not None:
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
                        point_in_body = do_transform_point(tgt_p_map, transform)

                        if point_in_body.point.x ** 2 + point_in_body.point.y ** 2 + point_in_body.point.z ** 2 < 1:
                            avd_vel = (0.0, 0.0, 0.0)
                        else:
                            target_direction = np.array([point_in_body.point.x, point_in_body.point.y, point_in_body.point.z])
                            # Normalize the target direction.
                            normalized_direction = target_direction / np.linalg.norm(target_direction)

                            # Convert normalized direction to yaw and pitch.
                            pitch_target = math.asin(normalized_direction[2])
                            yaw_target = math.atan2(normalized_direction[1], normalized_direction[0])

                            best_yaw, best_pitch, evade = vfh3d.target_direction(latest_obs, yaw_target, pitch_target, safety_distance=5, alpha=1.05)

                            hist = vfh3d.histogram[vfh3d.pitch_min_bin:vfh3d.pitch_max_bin+1, vfh3d.yaw_min_bin:vfh3d.yaw_max_bin+1][::-1, ::-1]*5
                            img = Image()
                            img.header.stamp = disp_ts
                            img.height = hist.shape[0]
                            img.width = hist.shape[1]
                            img.is_bigendian = 0
                            img.encoding = "mono8"
                            img.step = img.width
                            img.data = hist.astype(np.uint8).ravel()
                            hist_pub.publish(img)

                            costs = vfh3d.costs[vfh3d.pitch_min_bin:vfh3d.pitch_max_bin+1, vfh3d.yaw_min_bin:vfh3d.yaw_max_bin+1][::-1, ::-1]*10
                            img.data = costs.astype(np.uint8).ravel()
                            cost_pub.publish(img)

                            if best_yaw is None:
                                avd_vel = (0.0, 0.0, 0.0)
                                node.get_logger().warning('cannot find avoid vector', throttle_duration_sec = 0.1)
                            else:
                                # Convert spherical coordinates to a 3D vector.
                                y = math.cos(best_pitch) * math.sin(best_yaw)
                                z = math.sin(best_pitch)
                                if evade:
                                    v = np.array((0.0, y, z))
                                    avd_vel = v / np.linalg.norm(v)
                                else:
                                    x = math.cos(best_pitch) * math.cos(best_yaw)
                                    avd_vel = (x * 5, y * 5, z * 5)
                        m = TwistStamped()
                        m.header.frame_id = "body"
                        m.header.stamp = disp_ts
                        m.twist.linear.x = avd_vel[0]
                        m.twist.linear.y = avd_vel[1]
                        m.twist.linear.z = avd_vel[2]
                        avd_pub.publish(m)

                        latest_obs = None
        except KeyboardInterrupt:
            break
    rclpy.try_shutdown()
    print("bye")

if __name__ == '__main__':
    main()
