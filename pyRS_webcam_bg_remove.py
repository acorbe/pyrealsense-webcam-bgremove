## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
from scipy import ndimage
import pyfakewebcam


class BckCapacitor(object):
    def __init__(self):
        self.is_init = False

    def add_evidence(self, fg_mask, damping_coeff = 0.345):
        if not self.is_init:
            self.fg_mask = fg_mask.astype(np.float32)

        # damping
        self.fg_mask -= damping_coeff * self.fg_mask
        self.fg_mask += 0.1 * fg_mask.astype(np.float32)
        

    def get_mask(self):
        return self.fg_mask > 0.75

        
        


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

spatial = rs.spatial_filter()
spatial.set_option(rs.option.holes_fill, 3)
temporal = rs.temporal_filter()



# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

erode_kernel = np.ones((5,5),np.bool)


camera = pyfakewebcam.FakeWebcam('/dev/video13', 640, 480)


# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

try_alignment = True

bck_capacitor = BckCapacitor()


try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        if try_alignment:
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            # depth_image = np.asanyarray(aligned_depth_frame.get_data())
            # color_image = np.asanyarray(color_frame.get_data())
            depth_frame = aligned_depth_frame

        else:
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            

        filter_opt = False
        if filter_opt:
            depth_frame = spatial.process(depth_frame)

        temp_filter_opt = True
        if temp_filter_opt:
            depth_frame = temporal.process(depth_frame)
            

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())



        # print(np.max(depth_image), np.min(depth_image))


        # shadow = (depth_image == 0).astype(np.uint8)
        depth_mask_roi = ((depth_image > 0) & (depth_image < 1200)).astype(np.uint8)
        # depth_image[depth_mask_roi] = 0

        bck_capacitor.add_evidence(depth_mask_roi)

        final_image = color_image.copy()
        final_image = cv2.blur(final_image, (25,25)) # cv2.GaussianBlur(final_image, (5,5),cv2.BORDER_DEFAULT)
        cpy_blur = final_image.copy()

        # depth_mask_roi = cv2.erode(depth_mask_roi,erode_kernel,iterations = 1)

        depth_mask_roi = ndimage.binary_erosion(depth_mask_roi, structure=erode_kernel)

        depth_mask_roi_f = bck_capacitor.get_mask()
        final_image[depth_mask_roi_f] = color_image[depth_mask_roi_f]
        # final_image[depth_image == 0] = cpy_blur[depth_image == 0]

        color_image = final_image

        color_image = cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB)
        


        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # depth_colormap = depth_image

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape


        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

        camera.schedule_frame(color_image)

finally:

    # Stop streaming
    pipeline.stop()
