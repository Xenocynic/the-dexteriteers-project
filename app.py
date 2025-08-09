import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math
from PIL import Image

# Page configuration for full screen
st.set_page_config(
    page_title="Hand Games for Seniors",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar for full screen
)

# Custom CSS for full screen with floating controls
st.markdown("""
<style>
    /* Full screen layout */
    .main > div {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Floating control panel */
    .floating-controls {
        position: fixed;
        top: 20px;
        left: 20px;
        background: rgba(255, 255, 255, 0.95);
        border: 3px solid #4CAF50;
        border-radius: 15px;
        padding: 20px;
        z-index: 1000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        min-width: 200px;
    }
    
    /* Large, accessible buttons */
    .control-btn {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 15px 20px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        cursor: pointer;
        margin: 5px 0;
        width: 100%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    .control-btn:hover {
        background-color: #45a049;
    }
    
    .control-btn.stop {
        background-color: #f44336;
    }
    
    .control-btn.stop:hover {
        background-color: #da190b;
    }
    
    .control-btn.reset {
        background-color: #ff9800;
    }
    
    .control-btn.reset:hover {
        background-color: #e68900;
    }
    
    /* Status indicators */
    .status-item {
        margin: 8px 0;
        padding: 8px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    
    .status-good {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    /* Game selection */
    .game-selector {
        margin: 10px 0;
        font-size: 16px;
    }
    
    .game-selector select {
        font-size: 16px;
        padding: 8px;
        border-radius: 5px;
        border: 2px solid #4CAF50;
        width: 100%;
    }
    
    /* Score display */
    .score-display {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin: 10px 0;
        padding: 10px;
        background-color: #ecf0f1;
        border-radius: 8px;
        border: 2px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Performance tracking class
class PerformanceTracker:
    def __init__(self):
        self.frame_times = []
        self.start_time = time.time()
        self.total_frames = 0
        self.successful_detections = 0
        
    def add_frame_time(self, frame_time):
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        self.total_frames += 1
        
    def add_detection(self):
        self.successful_detections += 1
        
    def get_fps(self):
        if len(self.frame_times) < 2:
            return 0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0

# Game classes (same as before but optimized for full screen)
class Fruit:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = random.uniform(-6, -3)
        self.gravity = 0.25
        self.radius = 40  # Even larger for full screen
        self.color = random.choice([(255, 100, 100), (255, 200, 100), (100, 255, 100)])
        self.sliced = False
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
    
    def draw(self, frame):
        if not self.sliced:
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, self.color, -1)
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, (255, 255, 255), 4)

class SimplifiedFruitGame:
    def __init__(self):
        self.fruits = []
        self.score = 0
        self.last_spawn = time.time()
        self.spawn_interval = 2.0
        self.hand_trail = []
        self.trail_length = 8
    
    def spawn_fruit(self, width):
        if time.time() - self.last_spawn > self.spawn_interval:
            x = random.randint(100, width - 100)
            y = random.randint(500, 600)
            self.fruits.append(Fruit(x, y))
            self.last_spawn = time.time()
    
    def update_fruits(self, height):
        for fruit in self.fruits[:]:
            fruit.update()
            if fruit.y > height + 50:
                self.fruits.remove(fruit)
    
    def check_slicing(self, hand_x, hand_y):
        sliced_any = False
        for fruit in self.fruits:
            if not fruit.sliced:
                distance = math.sqrt((hand_x - fruit.x)**2 + (hand_y - fruit.y)**2)
                if distance < fruit.radius + 40:
                    fruit.sliced = True
                    self.score += 10
                    sliced_any = True
        return sliced_any
    
    def draw_fruits(self, frame):
        for fruit in self.fruits:
            if not fruit.sliced:
                fruit.draw(frame)
    
    def update_trail(self, hand_x, hand_y):
        self.hand_trail.append((hand_x, hand_y))
        if len(self.hand_trail) > self.trail_length:
            self.hand_trail.pop(0)
    
    def draw_trail(self, frame):
        if len(self.hand_trail) > 1:
            for i in range(1, len(self.hand_trail)):
                thickness = max(4, int((i / len(self.hand_trail)) * 12))
                cv2.line(frame, self.hand_trail[i-1], self.hand_trail[i], (0, 255, 255), thickness)

# Other game classes (simplified for space)
class Column:
    def __init__(self, x, screen_height, gap_size, width=100):
        self.x = x
        self.width = width
        self.gap_size = gap_size
        self.screen_height = screen_height
        self.top_height = random.randint(80, screen_height - gap_size - 80)
        self.bottom_y = self.top_height + gap_size
        self.bottom_height = screen_height - self.bottom_y

    def move(self, speed):
        self.x -= speed

    def draw(self, frame, color=(100, 200, 100), thickness=-1):
        cv2.rectangle(frame, (int(self.x), 0), (int(self.x + self.width), int(self.top_height)), color, thickness)
        cv2.rectangle(frame, (int(self.x), int(self.bottom_y)), (int(self.x + self.width), int(self.screen_height)), color, thickness)
        cv2.rectangle(frame, (int(self.x), 0), (int(self.x + self.width), int(self.top_height)), (255, 255, 255), 4)
        cv2.rectangle(frame, (int(self.x), int(self.bottom_y)), (int(self.x + self.width), int(self.screen_height)), (255, 255, 255), 4)

class FlappyBall:
    def __init__(self, radius=30):
        self.x = 150
        self.y = 300
        self.radius = radius

    def update_position(self, finger_y):
        self.y = int(finger_y)

    def draw(self, frame, color=(255, 100, 100)):
        cv2.circle(frame, (self.x, self.y), self.radius, color, -1)
        cv2.circle(frame, (self.x, self.y), self.radius, (255, 255, 255), 4)

class SimplifiedFlappyGame:
    def __init__(self, screen_height):
        self.ball = FlappyBall()
        self.columns = []
        self.score = 0
        self.last_column_spawn = time.time()
        self.screen_height = screen_height
        self.column_speed = 3

    def spawn_column(self, screen_width):
        if time.time() - self.last_column_spawn > 3.0:
            new_column = Column(x=screen_width, screen_height=self.screen_height, gap_size=200)
            self.columns.append(new_column)
            self.last_column_spawn = time.time()

    def update_columns(self):
        for column in self.columns[:]:
            column.move(self.column_speed)
            if column.x + column.width < 0:
                self.columns.remove(column)
                self.score += 1

    def check_collision(self):
        for column in self.columns:
            if column.x < self.ball.x < column.x + column.width:
                if self.ball.y - self.ball.radius < column.top_height or self.ball.y + self.ball.radius > column.bottom_y:
                    return True
        return False

    def update_ball(self, finger_y):
        self.ball.update_position(finger_y)

    def draw(self, frame):
        self.ball.draw(frame)
        for column in self.columns:
            column.draw(frame)

# Star Shooter classes (simplified)
class Star:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 40
        self.color = (100, 255, 255)

    def update(self, speed):
        self.y += speed

    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), self.size, self.color, -1)
        cv2.circle(frame, (int(self.x), int(self.y)), self.size, (255, 255, 255), 4)

class Bullet:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 10
        self.speed = 8
        self.color = (255, 255, 100)

    def update(self):
        self.y -= self.speed

    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius, self.color, -1)
        cv2.circle(frame, (int(self.x), int(self.y)), self.radius, (255, 255, 255), 3)

class Player:
    def __init__(self, screen_width, screen_height):
        self.x = screen_width // 2
        self.y = screen_height - 80
        self.radius = 30
        self.color = (100, 255, 100)

    def update_position(self, finger_x):
        self.x = int(finger_x)

    def draw(self, frame):
        cv2.circle(frame, (self.x, self.y), self.radius, self.color, -1)
        cv2.circle(frame, (self.x, self.y), self.radius, (255, 255, 255), 4)

class SimplifiedStarShooter:
    def __init__(self, screen_width, screen_height):
        self.player = Player(screen_width, screen_height)
        self.star = Star(random.randint(80, screen_width - 80), 80)
        self.bullets = []
        self.score = 0
        self.last_shot_time = time.time()
        self.shot_interval = 1.0
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.star_speed = 1.8

    def update(self):
        for bullet in self.bullets[:]:
            bullet.update()
            if bullet.y < 0:
                self.bullets.remove(bullet)
            else:
                distance = math.sqrt((bullet.x - self.star.x)**2 + (bullet.y - self.star.y)**2)
                if distance < self.star.size + bullet.radius:
                    self.score += 10
                    self.bullets.remove(bullet)
                    self.star = Star(random.randint(80, self.screen_width - 80), 80)

        self.star.update(self.star_speed)
        if self.star.y > self.screen_height:
            self.star = Star(random.randint(80, self.screen_width - 80), 80)

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

def main():
    # Initialize session state
    if 'perf_tracker' not in st.session_state:
        st.session_state.perf_tracker = PerformanceTracker()
    
    if 'game_start_time' not in st.session_state:
        st.session_state.game_start_time = None
    
    # Game duration in seconds (5 minutes)
    GAME_DURATION = 300
    
    # Create floating control panel HTML
    
    
    # Display floating controls
    
    
    # Full screen game display
    game_frame = st.empty()
    
    # Control inputs (hidden but functional)
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            game_choice = st.selectbox(
                "Game",
                ["Fruit Ninja", "Flappy Ball", "Star Shooter"],
                format_func=lambda x: (
                    f"🥷 {x}" if x == "Fruit Ninja" else
                    f"🏐 {x}" if x == "Flappy Ball" else
                    f"⭐ {x}"
                ),
                key="game_selector"
            )
        with col2:
            start_game = st.button("🟢 START", key="start_btn")
        with col3:
            reset_game = st.button("🔄 RESET", key="reset_btn")
        with col4:
            stop_game = st.button("🛑 STOP", key="stop_btn")
    
    # Initialize game
    if reset_game or 'game' not in st.session_state:
        if game_choice == "Fruit Ninja":
            st.session_state.game = SimplifiedFruitGame()
        elif game_choice == "Flappy Ball":
            st.session_state.game = SimplifiedFlappyGame(screen_height=720)
        else:
            st.session_state.game = SimplifiedStarShooter(screen_width=1280, screen_height=720)
        st.session_state.game_running = False
        st.session_state.perf_tracker = PerformanceTracker()
        st.session_state.current_game = game_choice
    
    # Check if game type changed
    if st.session_state.get('current_game') != game_choice:
        if game_choice == "Fruit Ninja":
            st.session_state.game = SimplifiedFruitGame()
        elif game_choice == "Flappy Ball":
            st.session_state.game = SimplifiedFlappyGame(screen_height=720)
        else:
            st.session_state.game = SimplifiedStarShooter(screen_width=1280, screen_height=720)
        st.session_state.current_game = game_choice
    
    if stop_game:
        st.session_state.game_running = False
    
    if start_game:
        st.session_state.game_running = True
        st.session_state.game_start_time = time.time()
    
    # Game loop
    if st.session_state.get('game_running', False):
        cap = cv2.VideoCapture(0)
        
        # Set camera for higher resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            st.error("❌ Camera not available")
            st.session_state.game_running = False
        else:
            with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.4,
                max_num_hands=1) as hands:
                
                frame_count = 0
                
                while st.session_state.get('game_running', False):
                    # Check timer
                    if st.session_state.game_start_time:
                        elapsed_time = time.time() - st.session_state.game_start_time
                        remaining_time = GAME_DURATION - elapsed_time
                        
                        if remaining_time <= 0:
                            st.session_state.game_running = False
                            break
                    
                    frame_start = time.time()
                    
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame = cv2.flip(frame, 1)
                    height, width, _ = frame.shape
                    
                    # Process hands
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb_frame)
                    
                    game = st.session_state.game
                    
                    # Game updates
                    if game_choice == "Fruit Ninja":
                        game.spawn_fruit(width)
                        game.update_fruits(height)
                    elif game_choice == "Flappy Ball":
                        game.spawn_column(width)
                        game.update_columns()
                    else:
                        game.update()
                    
                    hand_detected = False
                    game_over = False
                    
                    if results.multi_hand_landmarks:
                        hand_detected = True
                        st.session_state.perf_tracker.add_detection()
                        
                        for hand_landmarks in results.multi_hand_landmarks:
                            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            finger_x = int(index_tip.x * width)
                            finger_y = int(index_tip.y * height)
                            
                            if game_choice == "Fruit Ninja":
                                game.update_trail(finger_x, finger_y)
                                sliced = game.check_slicing(finger_x, finger_y)
                                color = (0, 255, 0) if sliced else (255, 255, 0)
                                cv2.circle(frame, (finger_x, finger_y), 20, color, -1)
                                cv2.circle(frame, (finger_x, finger_y), 20, (255, 255, 255), 4)
                                
                            elif game_choice == "Flappy Ball":
                                game.update_ball(finger_y)
                                cv2.circle(frame, (150, finger_y), 20, (0, 255, 0), -1)
                                cv2.circle(frame, (150, finger_y), 20, (255, 255, 255), 4)
                                if game.check_collision():
                                    game_over = True
                                    
                            else:
                                game.update_player(finger_x)
                                cv2.circle(frame, (finger_x, height - 80), 20, (0, 255, 0), -1)
                                cv2.circle(frame, (finger_x, height - 80), 20, (255, 255, 255), 4)
                    
                    # Draw games
                    if game_choice == "Fruit Ninja":
                        game.draw_trail(frame)
                        game.draw_fruits(frame)
                    elif game_choice == "Flappy Ball":
                        game.draw(frame)
                        if game_over:
                            cv2.putText(frame, "GAME OVER!", (width // 2 - 200, height // 2),
                                      cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
                            cv2.putText(frame, "GAME OVER!", (width // 2 - 200, height // 2),
                                      cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
                            time.sleep(2)
                            st.session_state.game_running = False
                    else:
                        game.draw(frame)
                    
                    # Draw HUD on screen
                    # Score (top center)
                    score_text = f"SCORE: {game.score}"
                    text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)[0]
                    score_x = (width - text_size[0]) // 2
                    cv2.putText(frame, score_text, (score_x, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)
                    cv2.putText(frame, score_text, (score_x, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
                    
                    # Timer (top right)
                    if st.session_state.game_start_time:
                        elapsed_time = time.time() - st.session_state.game_start_time
                        remaining_time = max(0, GAME_DURATION - elapsed_time)
                        minutes = int(remaining_time // 60)
                        seconds = int(remaining_time % 60)
                        timer_text = f"{minutes:02d}:{seconds:02d}"
                        
                        cv2.putText(frame, timer_text, (width - 200, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 6)
                        cv2.putText(frame, timer_text, (width - 200, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
                    
                    # Hand status (bottom right)
                    status_text = "HAND OK" if hand_detected else "SHOW HAND"
                    status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
                    cv2.putText(frame, status_text, (width - 250, height - 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)
                    cv2.putText(frame, status_text, (width - 250, height - 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 2)
                    
                    # Game instructions (bottom center)
                    if game_choice == "Fruit Ninja":
                        instruction = "Move finger to slice fruits"
                    elif game_choice == "Flappy Ball":
                        instruction = "Move finger up/down to guide ball"
                    else:
                        instruction = "Move finger left/right to aim"
                    
                    inst_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    inst_x = (width - inst_size[0]) // 2
                    cv2.putText(frame, instruction, (inst_x, height - 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
                    cv2.putText(frame, instruction, (inst_x, height - 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    
                    # Update display
                    game_frame.image(frame, channels="BGR", use_container_width=True)
                    
                    # Performance tracking
                    frame_time = time.time() - frame_start
                    st.session_state.perf_tracker.add_frame_time(frame_time)
                    frame_count += 1
                    
                    # Small delay
                    time.sleep(0.01)
        
        cap.release()
        st.session_state.game_running = False
    
    else:
        # Show start screen
        start_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        start_frame[:] = (50, 50, 50)  # Dark gray background
        
        # Welcome text
        welcome_text = "HAND TRACKING GAMES"
        text_size = cv2.getTextSize(welcome_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)[0]
        text_x = (1280 - text_size[0]) // 2
        cv2.putText(start_frame, welcome_text, (text_x, 200), 
                  cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        
        # Instructions
        instruction_text = f"Selected: {game_choice}"
        inst_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)[0]
        inst_x = (1280 - inst_size[0]) // 2
        cv2.putText(start_frame, instruction_text, (inst_x, 350), 
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 4)
        
        # Start prompt
        start_text = "Click START to begin your 5-minute session"
        start_size = cv2.getTextSize(start_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        start_x = (1280 - start_size[0]) // 2
        cv2.putText(start_frame, start_text, (start_x, 450), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
        
        # Tips
        tips = [
            "Sit 2-3 feet from camera",
            "Ensure good lighting", 
            "Move finger slowly and clearly",
            "Have fun and stay active!"
        ]
        
        for i, tip in enumerate(tips):
            tip_y = 550 + (i * 40)
            cv2.putText(start_frame, tip, (400, tip_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        
        game_frame.image(start_frame, channels="BGR", use_container_width=True)

if __name__ == "__main__":
    main()
