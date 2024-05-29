from hand import *
import cv2
import mediapipe as mp
import numpy as np
import random
from collections import defaultdict

class Box_Rect():
    def __init__(self, x, y, w, h, color, text='', transparency=0.5):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.text = text
        self.transparency = transparency
    
    def drawRect(self, img, text_color=(255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=2):
        alpha = self.transparency
        bg_rec = img[self.y: self.y + self.h, self.x: self.x + self.w].astype(np.float32)
        bg_rec_color = np.ones(bg_rec.shape, dtype=np.float32) * np.array(self.color, dtype=np.float32)
        res = cv2.addWeighted(bg_rec, alpha, bg_rec_color, 1-alpha, 0, dtype=cv2.CV_32F)
        if res is not None:
            img[self.y: self.y + self.h, self.x: self.x + self.w] = res.astype(np.uint8)
            text_size = cv2.getTextSize(self.text, font, font_scale, thickness)
            text_pos = (int(self.x + self.w/2 - text_size[0][0]/2), int(self.y + self.h/2 + text_size[0][1]/2))
            cv2.putText(img, self.text, text_pos, font, font_scale, text_color, thickness)

    def hand_paint(self, x, y):
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h


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

# Create canvas for drawing
drawing_canvas = np.zeros((720, 1280, 3), np.uint8)

# Initialize previous point for drawing
previous_x, previous_y = 0, 0

# Initial brush color and size
brush_color = (255, 0, 0)
brush_size = 5
eraser_size = 20

# Create color buttons
color_button = Box_Rect(200, 0, 100, 100, (120, 255, 0), 'Colors')

colors = [
    Box_Rect(300, 0, 100, 100, (0, 0, 255)),
    Box_Rect(400, 0, 100, 100, (255, 0, 0)),
    Box_Rect(500, 0, 100, 100, (0, 255, 0)),
    Box_Rect(600, 0, 100, 100, (255, 255, 0)),
    Box_Rect(700, 0, 100, 100, (255, 165, 0)),
    Box_Rect(800, 0, 100, 100, (128, 0, 128)),
    Box_Rect(900, 0, 100, 100, (255, 255, 255)),
    Box_Rect(1000, 0, 100, 100, (0, 0, 0), 'Eraser'),
    Box_Rect(1100, 0, 100, 100, (100, 100, 100), 'Clear'),
    Box_Rect(1200, 0, 100, 100, (255, 0, 0), 'Fill')
]

shape_buttons = [
    Box_Rect(1100, 100, 100, 100, (255, 255, 255), 'Circle'),
    Box_Rect(1100, 200, 100, 100, (255, 255, 255), 'Square'),
]


# Create canvas button
canvas_button = Box_Rect(50, 0, 100, 100, (0, 0, 255), 'Draw')
canvas = Box_Rect(50, 120, 1020, 580, (255, 255, 255), transparency=0.6)

cooldown_counter = 20
hide_canvas = True
hide_colors = True
hide_pen_sizes = True

            
while True:
    if cooldown_counter:
        cooldown_counter -= 1

    ret, frame = camera.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))
    frame = cv2.flip(frame, 1)

    hand_tracking.findHands(frame)
    positions = hand_tracking.getPostion(frame, draw=False)
    up_fingers = hand_tracking.getUpFingers(frame)

    if up_fingers:
        x, y = positions[8][0], positions[8][1]
        if up_fingers[1] and not canvas.hand_paint(x, y):
            previous_x, previous_y = 0, 0

            if not hide_colors:
                for shape_button in shape_buttons:
                    if shape_button.hand_paint(x, y):
                        selected_shape = shape_button.text
                        shape_button.transparency = 0
                        shape_locked = False
                        shape_position = None
                    else:
                        shape_button.transparency = 0.5

            # Choose color for drawing
            if not hide_colors:
                for cb in colors:
                    if cb.hand_paint(x, y):
                        brush_color = cb.color
                        cb.transparency = 0
                    else:
                        cb.transparency = 0.5

            # Toggle color panel visibility
            if color_button.hand_paint(x, y) and not cooldown_counter:
                cooldown_counter = 10
                color_button.transparency = 0
                hide_colors = not hide_colors
                color_button.text = 'Colors' if hide_colors else 'Close'
            else:
                color_button.transparency = 0.5


            # Toggle canvas visibility
            if canvas_button.hand_paint(x, y) and not cooldown_counter:
                cooldown_counter = 10
                canvas_button.transparency = 0
                hide_canvas = not hide_canvas
                canvas_button.text = 'Canvas' if hide_canvas else 'Close'

           # Initialize a separate canvas for filled shapes
            filled_canvas = np.zeros((720, 1280, 3), np.uint8)

            # Fill shape when hovering over the fill button
            if colors[-1].hand_paint(x, y) and not cooldown_counter:
                cooldown_counter = 10
                # Create a mask for the lines
                line_mask = np.zeros(drawing_canvas.shape[:2], dtype=np.uint8)
                gray_canvas = cv2.cvtColor(drawing_canvas, cv2.COLOR_BGR2GRAY)
                _, line_mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
                
                # Detect shape and fill
                contours, _ = cv2.findContours(cv2.cvtColor(drawing_canvas, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:
                        # Create a mask for the contour
                        mask = np.zeros(drawing_canvas.shape[:2], dtype=np.uint8)
                        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
                        
                        # Subtract the line mask from the contour mask
                        mask = cv2.subtract(mask, line_mask)
                        
                        # Create a colored image with the desired fill color
                        color_image = np.zeros_like(drawing_canvas)
                        color_image[:] = brush_color
                        
                        # Apply the mask to the colored image
                        filled_shape = cv2.bitwise_and(color_image, color_image, mask=mask)
                        
                        # Invert the mask
                        mask_inv = cv2.bitwise_not(mask)
                        
                        # Apply the inverted mask to the drawing canvas
                        drawing_canvas_bg = cv2.bitwise_and(drawing_canvas, drawing_canvas, mask=mask_inv)
                        
                        # Combine the filled shape with the drawing canvas
                        drawing_canvas = cv2.add(drawing_canvas_bg, filled_shape)

            # Move the drawing to the main image
            canvas_gray = cv2.cvtColor(drawing_canvas, cv2.COLOR_BGR2GRAY)
            _, inv_img = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY_INV)
            inv_img = cv2.cvtColor(inv_img, cv2.COLOR_GRAY2BGR)
            frame = cv2.bitwise_and(frame, inv_img)
            frame = cv2.bitwise_or(frame, drawing_canvas)

            # Clear canvas when hovering over the clear button
            if colors[-2].hand_paint(x, y) and not cooldown_counter:
                cooldown_counter = 10
                drawing_canvas = np.zeros((720, 1280, 3), np.uint8)

            # Set brush color to eraser color when hovering over the eraser button
            if colors[-3].hand_paint(x, y):
                brush_color = (0, 0, 0)
                colors[-3].transparency = 0
            else:
                colors[-3].transparency = 0.5

        elif up_fingers[1] and not up_fingers[2]:
            if canvas.hand_paint(x, y) and not hide_canvas:
                cv2.circle(frame, positions[8], brush_size, brush_color, -1)
                # Drawing on the canvas
                if previous_x == 0 and previous_y == 0:
                    previous_x, previous_y = positions[8]
                if brush_color == (0, 0, 0):
                    cv2.line(drawing_canvas, (previous_x, previous_y), positions[8], brush_color, eraser_size)
                else:
                    cv2.line(drawing_canvas, (previous_x, previous_y), positions[8], brush_color, brush_size)
                previous_x, previous_y = positions[8]
                
        elif up_fingers[1] and up_fingers[2]:
            if canvas.hand_paint(x, y) and not hide_canvas:
                if selected_shape == 'Circle' or selected_shape == 'Square':
                    if not shape_locked:
                        shape_position = positions[8]
                        shape_locked = True
                        initial_distance = np.sqrt((positions[8][0] - positions[12][0])**2 + (positions[8][1] - positions[12][1])**2)
                        initial_size = 30
                    else:
                        current_distance = np.sqrt((positions[8][0] - positions[12][0])**2 + (positions[8][1] - positions[12][1])**2)
                        zoom_factor = current_distance / initial_distance
                        current_size = int(initial_size * zoom_factor)
                        if selected_shape == 'Circle':
                            cv2.circle(drawing_canvas, shape_position, current_size, brush_color, -1)
                        elif selected_shape == 'Square':
                            shape_corner2 = (shape_position[0] + current_size, shape_position[1] + current_size)
                            cv2.rectangle(drawing_canvas, shape_position, shape_corner2, brush_color, -1)
                
                        

                else:
                    cv2.circle(frame, positions[8], brush_size, brush_color, -1)
                    # Drawing on the canvas
                    if previous_x == 0 and previous_y == 0:
                        previous_x, previous_y = positions[8]
                    if brush_color == (0, 0, 0):
                        cv2.line(drawing_canvas, (previous_x, previous_y), positions[8], brush_color, eraser_size)
                    else:
                        cv2.line(drawing_canvas, (previous_x, previous_y), positions[8], brush_color, brush_size)
                    previous_x, previous_y = positions[8]
        elif up_fingers[1] and not up_fingers[2]:
            if canvas.hand_paint(x, y) and not hide_canvas:
                if selected_shape == 'Circle' or selected_shape == 'Square':
                    if not shape_locked:
                        shape_position = positions[8]
                        shape_locked = True
                        initial_size = 30

                else:
                    cv2.circle(frame, positions[8], brush_size, brush_color, -1)
                    # Drawing on the canvas
                    if previous_x == 0 and previous_y == 0:
                        previous_x, previous_y = positions[8]
                    if brush_color == (0, 0, 0):
                        cv2.line(drawing_canvas, (previous_x, previous_y), positions[8], brush_color, eraser_size)
                    else:
                        cv2.line(drawing_canvas, (previous_x, previous_y), positions[8], brush_color, brush_size)
                    previous_x, previous_y = positions[8]

        else:
            previous_x, previous_y = 0, 0
            shape_locked = False

        

    # Draw color button
    color_button.drawRect(frame)
    cv2.rectangle(frame, (color_button.x, color_button.y), (color_button.x + color_button.w, color_button.y + color_button.h), (255, 255, 255), 2)

    # Draw canvas button
    canvas_button.drawRect(frame)
    cv2.rectangle(frame, (canvas_button.x, canvas_button.y), (canvas_button.x + canvas_button.w, canvas_button.y + canvas_button.h), (255, 255, 255), 2)

    # Draw canvas on the frame
    if not hide_canvas:
        canvas.drawRect(frame)
        # Move the drawing to the main image
        canvas_gray = cv2.cvtColor(drawing_canvas, cv2.COLOR_BGR2GRAY)
        _, inv_img = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY_INV)
        inv_img = cv2.cvtColor(inv_img, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, inv_img)
        frame = cv2.bitwise_or(frame, drawing_canvas)


        # Draw color boxes
    if not hide_colors:
        for c in colors:
            c.drawRect(frame)
            cv2.rectangle(frame, (c.x, c.y), (c.x + c.w, c.y + c.h), (255, 255, 255), 2)

    # Draw shape buttons
    if not hide_colors:
        for shape_button in shape_buttons:
            shape_button.drawRect(frame)
            cv2.rectangle(frame, (shape_button.x, shape_button.y), (shape_button.x + shape_button.w, shape_button.y + shape_button.h), (255, 255, 255), 2)

    cv2.imshow('Video', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()