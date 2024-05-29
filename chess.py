from hand import *
import cv2
import mediapipe as mp
import numpy as np
import random
from collections import defaultdict

class board():
    def __init__(self, text='', transparency=0.5):
        self.x = 50
        self.y = 50
        self.size = 640
        self.square = int(self.size/8)
        self.text = text
        self.transparency = transparency
        self.border = 5

        # Define starting positions
        center = (self.square/2, self.square/2)
        self.w_pawns = ["a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2"]

        # load piece images
        self.w_pawn = cv2.resize(cv2.imread("pieces/w_pawn.png", cv2.IMREAD_UNCHANGED), (self.square, self.square))
        self.b_pawn = cv2.resize(cv2.imread("pieces/b_pawn.png", cv2.IMREAD_UNCHANGED), (self.square, self.square))
        self.w_bish = cv2.resize(cv2.imread("pieces/w_bish.png", cv2.IMREAD_UNCHANGED), (self.square, self.square))
        self.b_bish = cv2.resize(cv2.imread("pieces/b_bish.png", cv2.IMREAD_UNCHANGED), (self.square, self.square))
        self.w_king = cv2.resize(cv2.imread("pieces/w_king.png", cv2.IMREAD_UNCHANGED), (self.square, self.square))
        self.b_king = cv2.resize(cv2.imread("pieces/b_king.png", cv2.IMREAD_UNCHANGED), (self.square, self.square))
        self.w_quee = cv2.resize(cv2.imread("pieces/w_quee.png", cv2.IMREAD_UNCHANGED), (self.square, self.square))
        self.b_quee = cv2.resize(cv2.imread("pieces/b_quee.png", cv2.IMREAD_UNCHANGED), (self.square, self.square))
        self.w_rook = cv2.resize(cv2.imread("pieces/w_rook.png", cv2.IMREAD_UNCHANGED), (self.square, self.square))
        self.b_rook = cv2.resize(cv2.imread("pieces/b_rook.png", cv2.IMREAD_UNCHANGED), (self.square, self.square))
        self.w_knight = cv2.resize(cv2.imread("pieces/w_knight.png", cv2.IMREAD_UNCHANGED), (self.square, self.square))
        self.b_knight = cv2.resize(cv2.imread("pieces/b_knight.png", cv2.IMREAD_UNCHANGED), (self.square, self.square))

    
    def draw(self, img):
        # Initialize blank mask image of same dimensions for drawing the shapes
        chessboard = np.zeros_like(img, dtype=np.uint8)
        color = (0, 0, 0)  # Start with black (0)

        # Iterate over the image in steps of blockSize
        for i in range(self.x, self.size+self.x, self.square):
            color = (0, 0, 0) if color == (255, 255, 255) else (255, 255, 255)
            for j in range(self.y, self.size+self.y, self.square):
                # set the board to the current color
                cv2.rectangle(chessboard, (i, j), (i+self.square, j+self.square), color, cv2.FILLED)

                # Toggle the color
                color = (0, 0, 0) if color == (255, 255, 255) else (255, 255, 255)

        # Generate output by blending image with shapes image, using the shapes
        # images also as mask to limit the blending to those parts
        out = img.copy()
        alpha = 0.5
        mask = chessboard.astype(bool)
        out[mask] = cv2.addWeighted(img, alpha, chessboard, 1 - alpha, 0)[mask]


        cv2.rectangle(out,
                      (self.x-self.border+3, self.y-self.border+3), 
                      (self.x+self.size+self.border-3, self.y+self.size+self.border-3),
                      (0,0,0), self.border)
    
        return out
    
def insert_image(background, overlay, point):
    x = point[0]
    y = point[1]
    h, w = overlay.shape[:2]

    # Split the overlay image into its channels
    b, g, r, a = cv2.split(overlay)

    # Create a mask and its inverse using the alpha channel
    alpha = a / 255.0
    alpha_inv = 1.0 - alpha

    # Determine the region of interest (ROI) on the background image
    roi = background[y:y + h, x:x + w]

    # Blend the overlay image onto the ROI
    for c in range(3):
        roi[:, :, c] = (alpha * overlay[:, :, c] + alpha_inv * roi[:, :, c])

    # Place the blended ROI back into the background image
    background[y:y + h, x:x + w] = roi

    return background


# Initialize hand detection module
hand_tracking = HandTracker(detectionCon=int(0.8))
selected_shape = None
shape_locked = False
initial_distance = 0
initial_size = 0
shape_position = None

# Initialize camera 
camera = cv2.VideoCapture(0)
camera.set(3, 1920)
camera.set(4, 1080)

# Initialize chess board
chessboard = board()

# Create canvas for chessboard
drawing_canvas = np.zeros((720, 720, 3), np.uint8)


w_pawn = cv2.resize(cv2.imread("pieces/w_pawn.png", cv2.IMREAD_UNCHANGED), (80, 80))




            
while True:

    ret, frame = camera.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))
    frame = cv2.flip(frame, 1)
    # TODO: Add in code

    res = chessboard.draw(frame)
    res = insert_image(res, w_pawn, (50, 50))

    cv2.imshow("Window", res)



    key = cv2.waitKey(1)
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()