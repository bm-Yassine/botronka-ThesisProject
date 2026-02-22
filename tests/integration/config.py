# Integration test hardware configuration.
# All pin/device constants live here so integration tests across the
# whole project share a single source of truth.

# ── Buzzer ────────────────────────────────────────────────────────────────────
BUZZER_PIN: int = 17

# ── Ultrasonic (HC-SR04) ──────────────────────────────────────────────────────
US_TRIG_GPIO: int = 23
US_ECHO_GPIO: int = 24

# ── OLED (SSD1306 I2C) ────────────────────────────────────────────────────────
I2C_BUS: int = 1
OLED_I2C_ADDR: int = 0x3C

# ── Motors (L298N) ────────────────────────────────────────────────────────────
MOTOR_ENA: int = 12
MOTOR_IN1: int = 5
MOTOR_IN2: int = 6
MOTOR_ENB: int = 13
MOTOR_IN3: int = 20
MOTOR_IN4: int = 21
