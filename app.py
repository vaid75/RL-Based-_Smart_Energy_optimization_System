from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_cors import CORS
from HEMS import HEMS
import matplotlib
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
matplotlib.use('Agg')
import numpy as np
import sqlite3
app = Flask(__name__)
app.secret_key = "secret123"
CORS(app)

# Initialize system with pre-trained model
system = HEMS(load=True, path="dqn_model.pth")


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login_page"


def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
  
    c.execute('SELECT COUNT(*) FROM users')
    if c.fetchone()[0] == 0:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', ('admin', '123'))
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', ('vaidehi', '456'))
    conn.commit()
    conn.close()

init_db()

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['username'])
        self.username = user_data['username']

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    user_data = conn.execute('SELECT * FROM users WHERE username = ?', (user_id,)).fetchone()
    conn.close()
    if user_data:
        return User(user_data)
    return None

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login_page"))

@app.route('/login_page')
def login_page():
    return render_template("login.html")

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template("index.html")

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data["username"]
    password = data["password"]

    conn = get_db_connection()
    user_data = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()

    if user_data and user_data['password'] == password:
        user = User(user_data)
        login_user(user)
        return jsonify({"status": "success"})
    
    return jsonify({"status": "invalid"})

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for("login_page"))


training_status = {
    "is_training": False,
    "error": None,
    "savings_trend": None,
    "message": ""
}

@app.route('/train', methods=['POST'])
@login_required
def train():
    global training_status
    if training_status["is_training"]:
        return jsonify({"status": "error", "message": "Training is already in progress"})

    try:
        episodes = int(request.json.get("episodes", 20))
        
        training_status["is_training"] = True
        training_status["error"] = None
        training_status["savings_trend"] = None
        training_status["message"] = "Training started..."

        def run_training():
            global training_status
            try:
                print("Training started for episodes:", episodes)
                _, episode_savings = system.train(n_episodes=episodes)
                print("Training finished")
                training_status["savings_trend"] = episode_savings
                training_status["message"] = "Training completed"
            except Exception as e:
                import traceback
                print("TRAIN ERROR:", traceback.format_exc())
                training_status["error"] = str(e)
                training_status["message"] = "Training failed"
            finally:
                training_status["is_training"] = False

        import threading
        t = threading.Thread(target=run_training)
        t.start()

        return jsonify({"status": "started", "message": "Training started in background"})
        
    except Exception as e:
        print("TRAIN START ERROR:", e)
        return jsonify({"status": "error", "message": str(e)})

@app.route('/train_status', methods=['GET'])
@login_required
def train_status_route():
    return jsonify(training_status)


@app.route('/test', methods=['POST'])
@login_required
def test():
    if training_status.get("is_training"):
        return jsonify({"error": "Training is currently in progress. Please wait until it finishes."})

    try:
        steps = int(request.json.get("steps", 200))
        steps = min(steps, 500)

        print("Running simulation for steps:", steps)
        result = system.test(steps=steps)
        print("Simulation completed")

        baseline = result.get("baseline_cost", 0)
        cost     = result.get("cost", 0)
        savings_pct = ((baseline - cost) / baseline * 100) if baseline > 0 else 0.0

        return jsonify({
            "cost":          round(cost, 4),
            "savings":       round(savings_pct, 2),
            "battery":       result["battery"],
            "solar_charge":  round(result.get("solar_charge", 0.0), 2),
            "sold_energy":   round(result.get("sold_energy", 0.0), 2),
            "rewards":       result["rewards"],
            "battery_levels": result["battery_levels"],
            "prices":        result["prices"],
        })

    except Exception as e:
        import traceback
        print("TEST ERROR:", traceback.format_exc())
        return jsonify({"error": str(e)})



if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
