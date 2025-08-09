import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math
from PIL import Image

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class Fruit:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-8, -4)
        self.gravity = 0.3
        self.radius = 30
        self.color = random.choice([(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0), (255, 0, 255)])
        self.sliced = False
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
    
    def draw(self, frame):
        if not self.sliced:
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, self.color, -1)
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, (0, 0, 0), 2)

class FruitNinjaGame:
    def __init__(self):
        self.fruits = []
        self.score = 0
        self.last_spawn = time.time()
        self.hand_trail = []
        self.trail_length = 10
    
    def spawn_fruit(self, width):
        if time.time() - self.last_spawn > 1.5:  # Spawn every 1.5 seconds
            x = random.randint(50, width - 50)
            y = random.randint(400, 500)
            self.fruits.append(Fruit(x, y))
            self.last_spawn = time.time()
    
    def update_fruits(self, height):
        for fruit in self.fruits[:]:
            fruit.update()
            if fruit.y > height + 100:  # Remove fruits that fall off screen
                self.fruits.remove(fruit)
    
    def check_slicing(self, hand_x, hand_y):
        for fruit in self.fruits:
            if not fruit.sliced:
                distance = math.sqrt((hand_x - fruit.x)**2 + (hand_y - fruit.y)**2)
                if distance < fruit.radius + 20:  # Collision detection
                    fruit.sliced = True
                    self.score += 10
                    return True
        return False
    
    def draw_fruits(self, frame):
        for fruit in self.fruits:
            fruit.draw(frame)
    
    def update_trail(self, hand_x, hand_y):
        self.hand_trail.append((hand_x, hand_y))
        if len(self.hand_trail) > self.trail_length:
            self.hand_trail.pop(0)
    
    def draw_trail(self, frame):
        for i in range(1, len(self.hand_trail)):
            thickness = int((i / len(self.hand_trail)) * 10)
            cv2.line(frame, self.hand_trail[i-1], self.hand_trail[i], (0, 255, 255), thickness)

import cv2
import random

class Column:
    def __init__(self, x, screen_height, gap_size, width=80):
        self.x = x
        self.width = width
        self.gap_size = gap_size
        self.screen_height = screen_height

        # Randomize the top column height
        self.top_height = random.randint(50, screen_height - gap_size - 50)
        self.bottom_y = self.top_height + gap_size
        self.bottom_height = screen_height - self.bottom_y

    def move(self, speed):
        self.x -= speed

    def draw(self, frame, color=(255, 0, 0), thickness=-1):
        # Draw top column
        top_left = (int(self.x), 0)
        bottom_right_top = (int(self.x + self.width), int(self.top_height))
        cv2.rectangle(frame, top_left, bottom_right_top, color, thickness)

        # Draw bottom column
        top_left_bottom = (int(self.x), int(self.bottom_y))
        bottom_right_bottom = (int(self.x + self.width), int(self.screen_height))
        cv2.rectangle(frame, top_left_bottom, bottom_right_bottom, color, thickness)

class FlappyBall:
    def __init__(self, radius=20):
        self.x = 100  # Fixed horizontal position
        self.y = 300  # Initial vertical position
        self.radius = radius

    def update_position(self, finger_y):
        self.y = int(finger_y)

    def draw(self, frame, color=(255, 0, 0)):
        cv2.circle(frame, (self.x, self.y), self.radius, color, -1)

class FlappyBallGame:
    def __init__(self, screen_height):
        self.ball = FlappyBall()
        self.columns = []
        self.score = 0
        self.last_column_spawn = time.time()
        self.screen_height = screen_height

    def spawn_column(self, screen_width):
        if time.time() - self.last_column_spawn > 1.7:
            new_column = Column(x=screen_width, screen_height=self.screen_height, gap_size=150)
            self.columns.append(new_column)
            self.last_column_spawn = time.time()

    def update_columns(self, speed=5):
        for column in self.columns[:]:
            column.move(speed)
            if column.x + column.width < 0:
                self.columns.remove(column)
                self.score += 1

    def check_collision(self):
        for column in self.columns:
            if column.x < self.ball.x < column.x + column.width:
                if self.ball.y < column.top_height or self.ball.y > column.bottom_y:
                    return True
        return False

    def update_ball(self, finger_y):
        self.ball.update_position(finger_y)

    def draw(self, frame):
        self.ball.draw(frame)
        for column in self.columns:
            column.draw(frame)
        cv2.putText(frame, f"Score: {self.score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


# ------------------- NEW STAR SHOOTER CLASSES -------------------
class Star:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 30
        self.color = (255, 255, 0)  # Yellow star

    def update(self, speed):
        self.y += speed  # Move down

    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), self.size, self.color, -1)
        cv2.circle(frame, (int(self.x), int(self.y)), self.size, (0, 0, 0), 2)


class Bullet:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 5
        self.speed = 10
        self.color = (0, 255, 255)  # Cyan bullet

    def update(self):
        self.y -= self.speed

    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius, self.color, -1)


class Player:
    def __init__(self, screen_width, screen_height):
        self.x = screen_width // 2
        self.y = screen_height - 50
        self.radius = 20
        self.color = (0, 255, 0)  # Green player

    def update_position(self, finger_x):
        self.x = int(finger_x)

    def draw(self, frame):
        cv2.circle(frame, (self.x, self.y), self.radius, self.color, -1)


class StarShooterGame:
    def __init__(self, screen_width, screen_height):
        self.player = Player(screen_width, screen_height)
        self.star = Star(random.randint(50, screen_width - 50), 50)
        self.bullets = []
        self.score = 0
        self.last_shot_time = time.time()
        self.shot_interval = 0.5  # Shoot every 0.5 seconds
        self.screen_width = screen_width
        self.screen_height = screen_height

    def update(self):
        # Update bullets
        for bullet in self.bullets[:]:
            bullet.update()
            if bullet.y < 0:
                self.bullets.remove(bullet)
            else:
                # Collision with star
                if ((bullet.x - self.star.x) ** 2 + (bullet.y - self.star.y) ** 2) ** 0.5 < self.star.size:
                    self.score += 10
                    self.bullets.remove(bullet)
                    self.star = Star(random.randint(50, self.screen_width - 50), 50)

        # Move star down
        self.star.update(2)
        if self.star.y > self.screen_height:
            self.star = Star(random.randint(50, self.screen_width - 50), 50)

        # Auto-fire
        if time.time() - self.last_shot_time > self.shot_interval:
            self.bullets.append(Bullet(self.player.x, self.player.y - self.player.radius))
            self.last_shot_time = time.time()

    def update_player(self, finger_x):
        self.player.update_position(finger_x)

    def draw(self, frame):
        self.player.draw(frame)
        self.star.draw(frame)
        for bullet in self.bullets:
            bullet.draw(frame)
        cv2.putText(frame, f"Score: {self.score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def main():
    st.title("🎮 Hand Tracking Games")
    
    # Game selection
    game_choice = st.selectbox(
        "Choose your game",
        ["Flappy Ball", "Fruit Ninja", "Star Shooter"],
        format_func=lambda x: (
            f"🏐 {x}" if x == "Flappy Ball" else
            f"🥷 {x}" if x == "Fruit Ninja" else
            f"⭐ {x}"
        ))
    
    # Initialize session state based on game choice
    if 'game' not in st.session_state or 'current_game' not in st.session_state or st.session_state.current_game != game_choice:
        st.session_state.current_game = game_choice
        if game_choice == "Flappy Ball":
            st.session_state.game = FlappyBallGame(screen_height=480)
        elif game_choice == "Fruit Ninja":
            st.session_state.game = FruitNinjaGame()
        else:
            st.session_state.game = StarShooterGame(screen_width=640, screen_height=480)
    
    # Display game instructions
    if game_choice == "Flappy Ball":
        st.write("Move your index finger up and down to guide the ball through the columns!")
    else:
        st.write("Move your hand to slice the falling fruits!")
    
    # Sidebar controls
    if st.sidebar.button("Reset Game"):
        if game_choice == "Flappy Ball":
            st.session_state.game = FlappyBallGame(screen_height=480)
        elif game_choice == "Fruit Ninja":
            st.session_state.game = FruitNinjaGame()
        else:
            st.session_state.game = StarShooterGame(screen_width=640, screen_height=480)

    
    confidence_threshold = st.sidebar.slider("Hand Detection Confidence", 0.1, 1.0, 0.7)
    
    # Game interface setup
    run = st.checkbox('Start Camera')
    FRAME_WINDOW = st.image([])
    score_placeholder = st.empty()
    
    if run:
        cap = cv2.VideoCapture(0)
        
        with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=0.5) as hands:
            
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access camera")
                    break
                
                frame = cv2.flip(frame, 1)
                height, width, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                game = st.session_state.game
                
                if game_choice == "Flappy Ball":
                    game.spawn_column(screen_width=width)
                    game.update_columns()
                elif game_choice == "Fruit Ninja":
                    game.spawn_fruit(width)
                    game.update_fruits(height)
                else:  # Star Shooter
                    game.update()
                
                # Hand tracking
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        finger_x = int(index_tip.x * width)
                        finger_y = int(index_tip.y * height)

                        if game_choice == "Flappy Ball":
                            game.update_ball(finger_y)
                            cv2.circle(frame, (100, finger_y), 10, (0, 255, 0), -1)
                        elif game_choice == "Fruit Ninja":
                            game.update_trail(finger_x, finger_y)
                            game.check_slicing(finger_x, finger_y)
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            cv2.circle(frame, (finger_x, finger_y), 10, (0, 255, 0), -1)
                        else:  # Star Shooter
                            game.update_player(finger_x)
                            cv2.circle(frame, (finger_x, height - 50), 10, (0, 255, 0), -1)

                
                # Game-specific updates
                if game_choice == "Flappy Ball":
                    if game.check_collision():
                        cv2.putText(frame, "Game Over!", (width // 2 - 100, height // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        cap.release()
                        break
                    game.draw(frame)
                elif game_choice == "Fruit Ninja":
                    game.draw_trail(frame)
                    game.draw_fruits(frame)
                    cv2.putText(frame, 'Point with index finger to slice!', (10, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:  # Star Shooter
                    game.draw(frame)

                
                # Display score and frame
                score_placeholder.metric("Score", game.score)
                FRAME_WINDOW.image(frame, channels="BGR")
        
        cap.release()

if __name__ == "__main__":
    main()
