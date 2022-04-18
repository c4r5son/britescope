from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN
import torch
import numpy as np
from imutils.video import FileVideoStream
import cv2

class FastMTCNN(object):
    """Fast MTCNN implementation."""
    
    def __init__(self, stride, resize=1, *args, **kwargs):
        """Constructor for FastMTCNN class.
        
        Arguments:
            stride (int): The detection stride. Faces will be detected every `stride` frames
                and remembered for `stride-1` frames.
        
        Keyword arguments:
            resize (float): Fractional frame scaling. [default: {1}]
            *args: Arguments to pass to the MTCNN constructor. See help(MTCNN).
            **kwargs: Keyword arguments to pass to the MTCNN constructor. See help(MTCNN).
        """
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)
        
    def __call__(self, frames):
        """Detect faces in frames using strided MTCNN."""
        if self.resize != 1:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                    for f in frames
            ]
                      
        return self.mtcnn.detect(frames[::self.stride])



def run_detection(frame):
    '''
    This method runs detection looking for faces using facenet
    @ret a tuple with (boxes to draw, probabilities a face was seen)
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    fast_mtcnn = FastMTCNN(
    stride=4,
    resize=1,
    margin=14,
    factor=0.5,
    keep_all=True,
    device=device
    )

    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frames = []
    frames.append(frame)
    boxes, prob  = fast_mtcnn(frames)

    return boxes, prob[0]


