from hand import *
import cv2
import numpy as np
import chess
import math


class board():
    def __init__(self, img, x=50, y=50, size=640):
        self.x = x
        self.y = y
        self.size = size
        self.square = int(self.size/8)
        self.border = 5
        self.selected = None

        # Initialize chess game api
        self.game = chess.Board()

        # create blank chessboard
        # Initialize blank mask image of same dimensions for drawing the shapes
        self.chessboard = np.zeros_like(img, dtype=np.uint8)
        color = (0, 0, 0)  # Start with black (0)

        # Iterate over the image in steps of blockSize
        for i in range(self.x, self.size+self.x, self.square):
            color = (0, 0, 0) if color == (255, 255, 255) else (255, 255, 255)
            for j in range(self.y, self.size+self.y, self.square):
                # set the board to the current color
                cv2.rectangle(self.chessboard, (i, j),
                              (i+self.square, j+self.square), color, cv2.FILLED)

                # Toggle the color
                color = (0, 0, 0) if color == (
                    255, 255, 255) else (255, 255, 255)

        cv2.rectangle(self.chessboard,
                      (self.x-self.border+3, self.y-self.border+3),
                      (self.x+self.size+self.border-3,
                       self.y+self.size+self.border-3),
                      (1, 1, 1), self.border)

        # load piece images
        self.image_dict = {
            # White pieces
            "P": cv2.resize(cv2.imread("pieces/w_pawn.png", cv2.IMREAD_UNCHANGED), (self.square, self.square)),
            "N": cv2.resize(cv2.imread("pieces/w_knight.png", cv2.IMREAD_UNCHANGED), (self.square, self.square)),
            "B": cv2.resize(cv2.imread("pieces/w_bish.png", cv2.IMREAD_UNCHANGED), (self.square, self.square)),
            "R": cv2.resize(cv2.imread("pieces/w_rook.png", cv2.IMREAD_UNCHANGED), (self.square, self.square)),
            "K": cv2.resize(cv2.imread("pieces/w_king.png", cv2.IMREAD_UNCHANGED), (self.square, self.square)),
            "Q": cv2.resize(cv2.imread("pieces/w_quee.png", cv2.IMREAD_UNCHANGED), (self.square, self.square)),

            # Black pieces
            "p": cv2.resize(cv2.imread("pieces/b_pawn.png", cv2.IMREAD_UNCHANGED), (self.square, self.square)),
            "n": cv2.resize(cv2.imread("pieces/b_knight.png", cv2.IMREAD_UNCHANGED), (self.square, self.square)),
            "b": cv2.resize(cv2.imread("pieces/b_bish.png", cv2.IMREAD_UNCHANGED), (self.square, self.square)),
            "r": cv2.resize(cv2.imread("pieces/b_rook.png", cv2.IMREAD_UNCHANGED), (self.square, self.square)),
            "k": cv2.resize(cv2.imread("pieces/b_king.png", cv2.IMREAD_UNCHANGED), (self.square, self.square)),
            "q": cv2.resize(cv2.imread("pieces/b_quee.png", cv2.IMREAD_UNCHANGED), (self.square, self.square))
        }

    def get_board(self, img):
        """
        Draw transparent chessboard.

        Args:
            img (cv2.Mat): Input image

        Returns:
            cv2.Mat: Image with chessboard
        """
        # Generate output by blending image with shapes image, using the shapes
        # images also as mask to limit the blending to those parts
        alpha = 0.4
        mask = self.chessboard.astype(bool)
        img[mask] = cv2.addWeighted(
            img, alpha, self.chessboard, 1 - alpha, 0)[mask]
        
        # if a square has been selected, make it yellow
        if self.selected:
            (x, y) = self.note_to_pix(self.selected)
            cv2.rectangle(img, (x, y), (x+self.square,
                            y+self.square), (0, 255, 255), -1)

        return img

    def note_to_pix(self, string):
        """
        Convert chess square notation to image pixels.

        Args:
            string (str): Chess notation

        Returns:
            Tuple(int, int): Image pixels
        """
        x = ord(string[0])-96-1
        y = 9-int(string[1])-1
        return (x*self.square+self.x, y*self.square+self.y)

    def pix_to_note(self, point):
        """
        Convert image pixels to chess square notation.

        Args:
            point (Tuple(int, int)): Pixel

        Returns:
            str: Chess notation
        """
        x = math.floor((point[0] - self.x)/self.square)
        y = math.floor((point[1] - self.y)/self.square)

        # convert to notation string
        return chr(x+97)+str(8-y)

    def draw_pieces(self, frame):
        """
        Draw chess pieces onto the frame.

        Args:
            frame (cv2.Mat): Image

        Returns:
            cv2.Mat: Image with the chess pieces drawn
        """
        # Get all the chess pieces
        for square in chess.SQUARES:
            piece = self.game.piece_at(square)
            if piece:
                # Get square name (e.g., 'e4') and convert to coordinates
                coords = self.note_to_pix(chess.square_name(square))
                piece_type = piece.symbol()  # Get piece symbol (e.g., 'P', 'n')
                frame = insert_image(frame,
                                     self.image_dict.get(piece_type),
                                     coords)

        return frame

    def action(self, square):
        """
        Calculates action based on selected square.

        Args:
            square (str): Chess square notation
        """
        # select square if none has been selected
        if not self.selected:
            # check if the square has a piece
            if not self.game.piece_at(chess.SQUARE_NAMES.index(square)):
                return
            self.selected = square
            return

        # if same square, no more actions
        if self.selected == square:
            return

        # if move is legal, make it and update the selected square
        move = chess.Move.from_uci(self.selected+square)
        if move in self.game.legal_moves:
            self.game.push(move)
            self.selected = None
            return

        # if none of the above, then update the selected square
        self.selected = square


def insert_image(background, overlay, point):
    """
    Insert an image onto the frame

    Args:
        background (cv2.Mat): Frame
        overlay (cv2.Mat): Image to insert
        point (Tuple(int, int)): Coordinate to insert the image

    Returns:
        cv2.Mat: The resulting image
    """
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


if __name__ == "__main__":
    # Initialize camera
    camera = cv2.VideoCapture(0)
    camera.set(3, 1920)
    camera.set(4, 1080)
    ret, frame = camera.read()

    # Initialize chess board
    x = 600
    y = 25
    size = 600
    chessboard = board(frame, x=x, y=y, size=size)

    # Initialze hand detector
    hand_tracking = HandTracker(maxHands=1, detectionCon=int(0.8))
    check = [True, True, False, False, False]

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Get the frame
        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.flip(frame, 1)

        # Find and draw hands
        hand_tracking.findHands(frame)

        # Chessboard interaction
        fingers = hand_tracking.getUpFingers(frame)

        # Check if only thumb and index finger is up
        if fingers == check:
            # Corresponds to index finger tip
            pos = hand_tracking.getPostion(frame, draw=False)[8]

            # If within board, select piece
            if x <= pos[0] <= x+size and y <= pos[1] <= y+size:
                # find the chess square
                note = chessboard.pix_to_note(pos)

                # Take action
                chessboard.action(note)

        # Draw chessboard and pieces
        res = chessboard.get_board(frame)
        chessboard.draw_pieces(res)

        cv2.imshow("CV Chess", res)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
