import React, { useEffect, useRef, useState } from "react";
import { Hands } from "@mediapipe/hands";
import { Camera } from "@mediapipe/camera_utils";

// Hand Games for Seniors - Full React refactor of the Streamlit/OpenCV app
// Single-file React component that uses MediaPipe Hands + HTML Canvas to render
// 3 games: Fruit Ninja, Flappy Ball, Star Shooter

// -------------------------- Utility / Performance --------------------------
class PerformanceTracker {
  constructor() {
    this.frameTimes = [];
    this.totalFrames = 0;
    this.successfulDetections = 0;
  }
  addFrameTime(dt) {
    this.frameTimes.push(dt);
    if (this.frameTimes.length > 60) this.frameTimes.shift();
    this.totalFrames += 1;
  }
  addDetection() {
    this.successfulDetections += 1;
  }
  getFps() {
    if (this.frameTimes.length < 2) return 0;
    const avg = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
    return avg > 0 ? 1.0 / avg : 0;
  }
}

// --------------------------------- Fruit ---------------------------------
class Fruit {
  constructor(x, y) {
    this.x = x;
    this.y = y;
    this.vx = Math.random() * 3 - 1.5;
    this.vy = Math.random() * -3 - 6;
    this.gravity = 0.25;
    this.radius = 40;
    this.color = ["#FF6464", "#FFC864", "#64FF64"][Math.floor(Math.random() * 3)];
    this.sliced = false;
  }
  update() {
    this.x += this.vx;
    this.y += this.vy;
    this.vy += this.gravity;
  }
  draw(ctx) {
    if (this.sliced) return;
    ctx.fillStyle = this.color;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 4;
    ctx.strokeStyle = "#FFFFFF";
    ctx.stroke();
  }
}

class SimplifiedFruitGame {
  constructor() {
    this.fruits = [];
    this.score = 0;
    this.lastSpawn = performance.now();
    this.spawnInterval = 2000; // ms
    this.handTrail = [];
    this.trailLength = 12;
  }
  spawnFruit(width) {
    const now = performance.now();
    if (now - this.lastSpawn > this.spawnInterval) {
      const x = Math.random() * (width - 200) + 100;
      const y = Math.random() * 80 + 500;
      this.fruits.push(new Fruit(x, y));
      this.lastSpawn = now;
    }
  }
  updateFruits(height) {
    this.fruits = this.fruits.filter((fruit) => {
      fruit.update();
      return fruit.y <= height + 50;
    });
  }
  checkSlicing(handX, handY) {
    let slicedAny = false;
    this.fruits.forEach((fruit) => {
      if (!fruit.sliced) {
        const dx = handX - fruit.x;
        const dy = handY - fruit.y;
        if (Math.sqrt(dx * dx + dy * dy) < fruit.radius + 40) {
          fruit.sliced = true;
          this.score += 10;
          slicedAny = true;
        }
      }
    });
    return slicedAny;
  }
  updateTrail(handX, handY) {
    if (handX == null || handY == null) return;
    this.handTrail.push({ x: handX, y: handY });
    if (this.handTrail.length > this.trailLength) this.handTrail.shift();
  }
  drawTrail(ctx) {
    if (this.handTrail.length < 2) return;
    for (let i = 1; i < this.handTrail.length; i++) {
      ctx.beginPath();
      ctx.moveTo(this.handTrail[i - 1].x, this.handTrail[i - 1].y);
      ctx.lineTo(this.handTrail[i].x, this.handTrail[i].y);
      ctx.lineWidth = Math.max(4, (i / this.handTrail.length) * 12);
      ctx.strokeStyle = "rgba(0,255,255,0.9)";
      ctx.stroke();
    }
  }
  drawFruits(ctx) {
    this.fruits.forEach((f) => f.draw(ctx));
  }
}

// -------------------------------- Flappy --------------------------------
class Column {
  constructor(x, screenHeight, gapSize, width = 100) {
    this.x = x;
    this.width = width;
    this.gapSize = gapSize;
    this.screenHeight = screenHeight;
    this.topHeight = Math.floor(Math.random() * (screenHeight - gapSize - 160)) + 80;
    this.bottomY = this.topHeight + gapSize;
  }
  move(speed) {
    this.x -= speed;
  }
  draw(ctx, color = "#64C864") {
    ctx.fillStyle = color;
    ctx.fillRect(this.x, 0, this.width, this.topHeight);
    ctx.fillRect(this.x, this.bottomY, this.width, this.screenHeight - this.bottomY);
    ctx.lineWidth = 4;
    ctx.strokeStyle = "#FFFFFF";
    ctx.strokeRect(this.x, 0, this.width, this.topHeight);
    ctx.strokeRect(this.x, this.bottomY, this.width, this.screenHeight - this.bottomY);
  }
}
class FlappyBall {
  constructor(radius = 30) {
    this.x = 150;
    this.y = 300;
    this.radius = radius;
  }
  updatePosition(fingerY) {
    this.y = Math.max(this.radius + 10, Math.min(720 - this.radius - 10, Math.floor(fingerY)));
  }
  draw(ctx, color = "#FF6464") {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 4;
    ctx.strokeStyle = "#FFFFFF";
    ctx.stroke();
  }
}
class SimplifiedFlappyGame {
  constructor(screenHeight) {
    this.ball = new FlappyBall();
    this.columns = [];
    this.score = 0;
    this.lastColumnSpawn = performance.now();
    this.screenHeight = screenHeight;
    this.columnSpeed = 3.0;
    this.gapSize = 200;
  }
  spawnColumn(screenWidth) {
    if (performance.now() - this.lastColumnSpawn > 3000) {
      this.columns.push(new Column(screenWidth, this.screenHeight, this.gapSize));
      this.lastColumnSpawn = performance.now();
    }
  }
  updateColumns() {
    for (let i = this.columns.length - 1; i >= 0; i--) {
      const c = this.columns[i];
      c.move(this.columnSpeed);
      if (c.x + c.width < 0) {
        this.columns.splice(i, 1);
        this.score += 1;
      }
    }
  }
  checkCollision() {
    for (const column of this.columns) {
      if (column.x < this.ball.x && this.ball.x < column.x + column.width) {
        if (this.ball.y - this.ball.radius < column.topHeight || this.ball.y + this.ball.radius > column.bottomY) {
          return true;
        }
      }
    }
    return false;
  }
  updateBall(fingerY) {
    this.ball.updatePosition(fingerY);
  }
  draw(ctx) {
    this.ball.draw(ctx);
    this.columns.forEach((c) => c.draw(ctx));
  }
}

// -------------------------------- Star Shooter ----------------------------
class Star {
  constructor(x, y) {
    this.x = x;
    this.y = y;
    this.size = 40;
    this.color = "#64FFFF";
  }
  update(speed) {
    this.y += speed;
  }
  draw(ctx) {
    ctx.fillStyle = this.color;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 4;
    ctx.strokeStyle = "#FFFFFF";
    ctx.stroke();
  }
}
class Bullet {
  constructor(x, y) {
    this.x = x;
    this.y = y;
    this.radius = 10;
    this.speed = 8;
    this.color = "#FFFF64";
  }
  update() {
    this.y -= this.speed;
  }
  draw(ctx) {
    ctx.fillStyle = this.color;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#FFFFFF";
    ctx.stroke();
  }
}
class Player {
  constructor(screenWidth, screenHeight) {
    this.x = Math.floor(screenWidth / 2);
    this.y = screenHeight - 80;
    this.radius = 30;
    this.color = "#64FF64";
  }
  updatePosition(fingerX) {
    this.x = Math.max(40, Math.min(this.x, fingerX | 0));
  }
  draw(ctx) {
    ctx.fillStyle = this.color;
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 4;
    ctx.strokeStyle = "#FFFFFF";
    ctx.stroke();
  }
}
class SimplifiedStarShooter {
  constructor(screenWidth, screenHeight) {
    this.player = new Player(screenWidth, screenHeight);
    this.star = new Star(Math.random() * (screenWidth - 160) + 80, 80);
    this.bullets = [];
    this.score = 0;
    this.lastShotTime = performance.now();
    this.shotInterval = 1000; // ms
    this.screenWidth = screenWidth;
    this.screenHeight = screenHeight;
    this.starSpeed = 1.8;
  }
  update() {
    for (let i = this.bullets.length - 1; i >= 0; i--) {
      const b = this.bullets[i];
      b.update();
      if (b.y < 0) this.bullets.splice(i, 1);
      else {
        const dx = b.x - this.star.x;
        const dy = b.y - this.star.y;
        if (Math.sqrt(dx * dx + dy * dy) < this.star.size + b.radius) {
          this.score += 10;
          this.bullets.splice(i, 1);
          this.star = new Star(Math.random() * (this.screenWidth - 160) + 80, 80);
        }
      }
    }
    this.star.update(this.starSpeed);
    if (this.star.y > this.screenHeight) this.star = new Star(Math.random() * (this.screenWidth - 160) + 80, 80);

    if (performance.now() - this.lastShotTime > this.shotInterval) {
      this.bullets.push(new Bullet(this.player.x, this.player.y - this.player.radius));
      this.lastShotTime = performance.now();
    }
  }
  updatePlayer(fingerX) {
    if (fingerX != null) this.player.x = Math.max(40, Math.min(this.screenWidth - 40, Math.floor(fingerX)));
  }
  draw(ctx) {
    this.player.draw(ctx);
    this.star.draw(ctx);
    this.bullets.forEach((b) => b.draw(ctx));
  }
}

// -------------------------------- React App --------------------------------
export default function HandGamesApp() {
  const canvasRef = useRef(null);
  const videoRef = useRef(null);
  const cameraRef = useRef(null);
  const handsRef = useRef(null);
  const perfRef = useRef(new PerformanceTracker());
  const gameRef = useRef(null);
  const rafRef = useRef(null);

  const [gameChoice, setGameChoice] = useState("Fruit Ninja");
  const [running, setRunning] = useState(false);
  const [score, setScore] = useState(0);
  const [remaining, setRemaining] = useState(300); // seconds
  const [startTime, setStartTime] = useState(null);
  const [handOk, setHandOk] = useState(false);

  // initialize game instance when choice changes or reset
  function initGame(choice) {
    const width = 1280;
    const height = 720;
    if (choice === "Fruit Ninja") gameRef.current = new SimplifiedFruitGame();
    else if (choice === "Flappy Ball") gameRef.current = new SimplifiedFlappyGame(height);
    else gameRef.current = new SimplifiedStarShooter(width, height);
    setScore(0);
  }

  useEffect(() => {
    initGame(gameChoice);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gameChoice]);

  // Setup MediaPipe Hands and Camera on mount
  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });
    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 0,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.4,
    });

    hands.onResults(async (results) => {
      const frameStart = performance.now();

      // draw background (camera frame)
      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (results.image) ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

      // update game behaviour
      const g = gameRef.current;
      if (!g) {
        ctx.restore();
        return;
      }

      // spawn/update depending on game
      if (gameChoice === "Fruit Ninja") {
        g.spawnFruit(canvas.width);
        g.updateFruits(canvas.height);
      } else if (gameChoice === "Flappy Ball") {
        g.spawnColumn(canvas.width);
        g.updateColumns();
      } else {
        g.update();
      }

      let handDetectedNow = false;
      let fingerX = null;
      let fingerY = null;

      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        handDetectedNow = true;
        perfRef.current.addDetection();
        // take first hand, index finger tip (landmark 8)
        const lm = results.multiHandLandmarks[0][8];
        fingerX = lm.x * canvas.width;
        fingerY = lm.y * canvas.height;

        if (gameChoice === "Fruit Ninja") {
          g.updateTrail(fingerX, fingerY);
          g.checkSlicing(fingerX, fingerY);
          // draw finger marker
          ctx.beginPath();
          ctx.arc(fingerX, fingerY, 20, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(0,255,0,0.9)";
          ctx.fill();
          ctx.lineWidth = 4;
          ctx.strokeStyle = "#FFFFFF";
          ctx.stroke();
        } else if (gameChoice === "Flappy Ball") {
          g.updateBall(fingerY);
          ctx.beginPath();
          ctx.arc(150, fingerY, 20, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(0,255,0,0.9)";
          ctx.fill();
          ctx.lineWidth = 4;
          ctx.strokeStyle = "#FFFFFF";
          ctx.stroke();
        } else {
          g.updatePlayer(fingerX);
          ctx.beginPath();
          ctx.arc(fingerX, canvas.height - 80, 20, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(0,255,0,0.9)";
          ctx.fill();
          ctx.lineWidth = 4;
          ctx.strokeStyle = "#FFFFFF";
          ctx.stroke();
        }
      }

      // draw game visuals
      if (gameChoice === "Fruit Ninja") {
        g.drawTrail(ctx);
        g.drawFruits(ctx);
      } else if (gameChoice === "Flappy Ball") {
        g.draw(ctx);
        if (g.checkCollision()) {
          // show game over text overlay briefly
          ctx.font = "bold 72px Arial";
          ctx.fillStyle = "#FF0000";
          ctx.fillText("GAME OVER!", canvas.width / 2 - 220, canvas.height / 2);
          // stop running by setting running false in next tick
          setTimeout(() => setRunning(false), 500);
        }
      } else {
        g.draw(ctx);
      }

      // HUD: score
      ctx.font = "bold 36px Arial";
      ctx.fillStyle = "#FFFFFF";
      ctx.lineWidth = 6;
      const scoreText = `SCORE: ${g.score}`;
      const metrics = ctx.measureText(scoreText);
      ctx.fillText(scoreText, (canvas.width - metrics.width) / 2, 60);

      // timer
      if (startTime) {
        const elapsed = Math.floor((performance.now() - startTime) / 1000);
        const rem = Math.max(0, 300 - elapsed);
        setRemaining(rem);
        const mm = String(Math.floor(rem / 60)).padStart(2, "0");
        const ss = String(rem % 60).padStart(2, "0");
        const timerText = `${mm}:${ss}`;
        ctx.font = "bold 28px Arial";
        ctx.fillText(timerText, canvas.width - 160, 60);
        if (rem <= 0) setRunning(false);
      }

      // status
      ctx.font = "20px Arial";
      ctx.fillStyle = handDetectedNow ? "#00FF00" : "#FF3333";
      ctx.fillText(handDetectedNow ? "HAND OK" : "SHOW HAND", canvas.width - 240, canvas.height - 30);

      // instructions
      ctx.font = "22px Arial";
      ctx.fillStyle = "#FFFFFF";
      const instr = gameChoice === "Fruit Ninja" ? "Move finger to slice fruits" : gameChoice === "Flappy Ball" ? "Move finger up/down to guide ball" : "Move finger left/right to aim";
      const instMetrics = ctx.measureText(instr);
      ctx.fillText(instr, (canvas.width - instMetrics.width) / 2, canvas.height - 30);

      ctx.restore();

      const frameTime = performance.now() - frameStart;
      perfRef.current.addFrameTime(frameTime / 1000);
      setHandOk(handDetectedNow);
      setScore(g.score);
    });

    handsRef.current = hands;

    const camera = new Camera(video, {
      onFrame: async () => {
        await hands.send({ image: video });
      },
      width: 1280,
      height: 720,
    });
    cameraRef.current = camera;

    return () => {
      if (cameraRef.current) cameraRef.current.stop();
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [startTime, gameChoice]);

  // Start / Stop handlers
  function handleStart() {
    setRunning(true);
    setStartTime(performance.now());
    setRemaining(300);
    perfRef.current = new PerformanceTracker();
    if (cameraRef.current) cameraRef.current.start();
  }
  function handleStop() {
    setRunning(false);
    if (cameraRef.current) cameraRef.current.stop();
  }
  function handleReset() {
    initGame(gameChoice);
    setScore(0);
    setStartTime(null);
    setRemaining(300);
  }

  // When running toggles, ensure camera is started/stopped
  useEffect(() => {
    if (running) {
      if (cameraRef.current) cameraRef.current.start();
      setStartTime((s) => (s ? s : performance.now()));
    } else {
      if (cameraRef.current) cameraRef.current.stop();
    }
  }, [running]);

  // set canvas/video sizes once
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    canvas.width = 1280;
    canvas.height = 720;
    video.width = 1280;
    video.height = 720;
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center relative">
      {/* Floating Controls */}
      <div className="floating-controls fixed top-6 left-6 bg-white/95 border-4 border-green-500 rounded-2xl p-6 shadow-lg z-50 min-w-[240px]">
        <div className="mb-3">
          <label className="block font-bold text-gray-700 mb-2">Game</label>
          <select
            aria-label="Select game"
            className="w-full p-2 rounded-md border-2 border-green-500"
            value={gameChoice}
            onChange={(e) => {
              setGameChoice(e.target.value);
              initGame(e.target.value);
            }}
          >
            <option>Fruit Ninja</option>
            <option>Flappy Ball</option>
            <option>Star Shooter</option>
          </select>
        </div>

        <div className="flex flex-col gap-2">
          <button className="control-btn bg-green-500 text-white font-bold py-3 rounded-lg" onClick={handleStart}>
            🟢 START
          </button>
          <button className="control-btn reset bg-orange-400 text-white font-bold py-3 rounded-lg" onClick={handleReset}>
            🔄 RESET
          </button>
          <button className="control-btn stop bg-red-500 text-white font-bold py-3 rounded-lg" onClick={handleStop}>
            🛑 STOP
          </button>
        </div>

        <div className="mt-4 p-2 bg-gray-100 rounded-md text-center">
          <div className="font-bold text-lg">SCORE</div>
          <div className="text-2xl">{score}</div>
        </div>

        <div className="mt-3 p-2 bg-gray-100 rounded-md text-center">
          <div className="font-bold">TIMER</div>
          <div>{String(Math.floor(remaining / 60)).padStart(2, "0")}:{String(remaining % 60).padStart(2, "0")}</div>
        </div>

        <div className="mt-3 p-2 rounded-md text-center">
          <div className={`status-item ${handOk ? "status-good" : "status-error"}`}>
            {handOk ? "HAND OK" : "SHOW HAND"}
          </div>
          <div className="text-sm text-gray-500 mt-2">FPS: {Math.round(perfRef.current.getFps())}</div>
        </div>
      </div>

      {/* Hidden video used as MediaPipe input */}
      <video ref={videoRef} style={{ display: "none" }} playsInline />

      {/* Main canvas */}
      <canvas ref={canvasRef} style={{ maxWidth: "100%", height: "auto" }} />

      <style>{`
        .floating-controls .control-btn { cursor: pointer; }
        .status-item { padding: 8px; border-radius: 6px; font-weight: bold; }
        .status-good { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status-error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
      `}</style>
    </div>
  );
}
