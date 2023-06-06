import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

class CameraPublisherNode(Node):
    def __init__(self):
        super().__init__('camera_publisher_node')
        self.publisher_ = self.create_publisher(Image, 'camera_image', 10)
        self.bridge = CvBridge()

    def publish_image(self):
        cap = cv2.VideoCapture(0)  # Access camera

        # Create a GestureRecognizer object
        base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
        options = vision.GestureRecognizerOptions(base_options=base_options)
        recognizer = vision.GestureRecognizer.create_from_options(options)

        while True:
            ret, frame = cap.read()

            if ret:
                # Convert the frame to ROS Image message
                img_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')

                # Publish the image message
                self.publisher_.publish(img_msg)

                # Display the video stream in an OpenCV window
                cv2.imshow('Camera Stream', frame)

                # Convert the frame to MediaPipe image format
                mp_frame = mp.Image(np.uint8(frame))

                # Recognize gestures in the input image
                recognition_result = recognizer.recognize(mp_frame)

                # Process the result. In this case, visualize it.
                top_gesture = recognition_result.gestures[0][0]
                hand_landmarks = recognition_result.hand_landmarks

                print("Detected gesture:", top_gesture)

            if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit the loop
                break

        cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherNode()
    node.publish_image()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
