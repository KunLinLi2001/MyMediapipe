# 导入相关模块
import enum
from typing import NamedTuple
import numpy as np
from mediapipe.calculators.core import constant_side_packet_calculator_pb2
from mediapipe.calculators.core import gate_calculator_pb2
from mediapipe.calculators.core import split_vector_calculator_pb2
from mediapipe.calculators.tensor import image_to_tensor_calculator_pb2
from mediapipe.calculators.tensor import inference_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_classification_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_detections_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_landmarks_calculator_pb2
from mediapipe.calculators.tflite import ssd_anchors_calculator_pb2
from mediapipe.calculators.util import association_calculator_pb2
from mediapipe.calculators.util import detections_to_rects_calculator_pb2
from mediapipe.calculators.util import logic_calculator_pb2
from mediapipe.calculators.util import non_max_suppression_calculator_pb2
from mediapipe.calculators.util import rect_transformation_calculator_pb2
from mediapipe.calculators.util import thresholding_calculator_pb2
# 引入了HandConnections模块作为一个手部各关节点构成的列表
from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS
from mediapipe.python.solution_base import SolutionBase

# HandLandmark类作为枚举类型，包括21个手部关节点
class HandLandmark(enum.IntEnum):
  WRIST = 0
  THUMB_CMC = 1
  THUMB_MCP = 2
  THUMB_IP = 3
  THUMB_TIP = 4
  INDEX_FINGER_MCP = 5
  INDEX_FINGER_PIP = 6
  INDEX_FINGER_DIP = 7
  INDEX_FINGER_TIP = 8
  MIDDLE_FINGER_MCP = 9
  MIDDLE_FINGER_PIP = 10
  MIDDLE_FINGER_DIP = 11
  MIDDLE_FINGER_TIP = 12
  RING_FINGER_MCP = 13
  RING_FINGER_PIP = 14
  RING_FINGER_DIP = 15
  RING_FINGER_TIP = 16
  PINKY_MCP = 17
  PINKY_PIP = 18
  PINKY_DIP = 19
  PINKY_TIP = 20

# mediapipe的hand_landmark_tracking_cpu模型的文件路径
_BINARYPB_FILE_PATH = 'mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.binarypb'

# Hands类继承自SolutionBase类, 使用手部关键点检测
class Hands(SolutionBase):
  def __init__(self, static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    # 在SolutionBase中对输入和输出进行了定义， 其余参数在此初始化。最终调用SolutionBase构造函数。
    super().__init__(
        binary_graph_path=_BINARYPB_FILE_PATH,
        side_inputs={
            'model_complexity': model_complexity, # 模型复杂度
            'num_hands': max_num_hands, # 最大手掌检测数目
            'use_prev_landmarks': not static_image_mode, # 是否使用之前的手掌信息
        },
        calculator_params={
            'palmdetectioncpu__TensorsToDetectionsCalculator.min_score_thresh': # 手掌检测的最小分数阈值. 被认为是有效手的最低置信值
                min_detection_confidence,
            'handlandmarkcpu__ThresholdingCalculator.threshold': # 最小关键点置信值.被判定为有效的关键点的最低置信阈值
                min_tracking_confidence,
        },
        outputs=[
            'multi_hand_landmarks', # 手部关键点列表 对于每个检测到的手
            'multi_hand_world_landmarks', # 当前帧的手部3D坐标列表，参考坐标系为手掌的重心
            'multi_handedness' # 分别表示对应检测到手的左/右手
        ]
    )

  def process(self, image: np.ndarray) -> NamedTuple:
    """处理帧图片，并返回检测到的手的关键点和对应的左右手
    Args:
      image: 用numpy.ndarray表示的RGB图片
    Raises:
      RuntimeError: 如果图形处理出现意外问题
      ValueError: 如果图像不是3通道的RGB格式

    Returns:
      用以下字段命名元组返回结果:
        1) A "multi_hand_landmarks" field that contains the hand landmarks on
           each detected hand.
        2) A "multi_hand_world_landmarks" field that contains the hand landmarks
           on each detected hand in real-world 3D coordinates that are in meters
           with the origin at the hand's approximate geometric center.
        3) A "multi_handedness" field that contains the handedness (left v.s.
           right hand) of the detected hand.
    """
    return super().process(input_data={'image': image})
