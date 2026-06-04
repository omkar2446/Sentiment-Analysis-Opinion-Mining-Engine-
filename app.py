

import os
import pickle
import datetime
import sqlite3
from flask import Flask, render_template, request, jsonify, redirect, url_for
import mysql.connector
from mysql.connector import pooling

app = Flask(__name__)

# --- CONFIGURATION ---
# Database configuration using environment variables with local defaults.
DB_CONFIG = {
    'host': os.environ.get('MYSQL_HOST', 'localhost'),
    'user': os.environ.get('MYSQL_USER', 'root'),
    'password': os.environ.get('MYSQL_PASSWORD', ''),
    'database': os.environ.get('MYSQL_DATABASE', 'sentiment_db'),
    'port': int(os.environ.get('MYSQL_PORT', 3306))
}

# Database State
db_type = 'mysql'      # Will switch to 'sqlite' if MySQL connection fails
db_error = None        # Stores MySQL error messages to display in the UI banner
db_pool = None

# --- MODEL LOADING ---
MODEL_PATH = 'sentiment_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

model = None
vectorizer = None

def load_models():
    """Loads the pre-trained model and vectorizer pickle files."""
    global model, vectorizer
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        print("Model and Vectorizer loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: Model or Vectorizer file not found. {e}")
        print("Please run 'python train_model.py' to generate model and vectorizer files first.")
    except Exception as e:
        print(f"Error loading models: {e}")

# Load models immediately on app load
load_models()

# --- DATABASE SETUP & CONNECTIONS ---

def init_db():
    """
    Initializes the database.
    Attempts to connect and configure MySQL. If MySQL connection fails, 
    falls back to a local SQLite database so the application remains functional.
    """
    global db_pool, db_type, db_error
    try:
        # Step 1: Connect to MySQL server to ensure database exists
        server_conn = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            port=DB_CONFIG['port']
        )
        server_cursor = server_conn.cursor()
        server_cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        server_cursor.close()
        server_conn.close()

        # Step 2: Connect to database to verify tables
        db_conn = mysql.connector.connect(**DB_CONFIG)
        db_cursor = db_conn.cursor()
        
        # Predictions table
        db_cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                review_text TEXT NOT NULL,
                predicted_sentiment VARCHAR(15) NOT NULL,
                confidence FLOAT DEFAULT 0.0,
                prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        db_conn.commit()
        db_cursor.close()
        db_conn.close()

        # Step 3: Establish connection pool for app threads
        db_pool = pooling.MySQLConnectionPool(
            pool_name="sentix_pool",
            pool_size=5,
            pool_reset_session=True,
            **DB_CONFIG
        )
        db_type = 'mysql'
        print("MySQL connection pool established successfully!")
    except mysql.connector.Error as err:
        db_type = 'sqlite'
        db_error = (
            f"MySQL Connection Failed: {err}. "
            "Falling back to local SQLite database 'sentiment_db.sqlite' for demo/testing. "
            "To connect to MySQL, update your credentials in app.py DB_CONFIG."
        )
        print("\n" + "="*80)
        print("DATABASE WARNING: Unable to connect to MySQL database.")
        print(f"Details: {err}")
        print("FALLING BACK TO SQLITE FOR RUNNING THE APP.")
        print("="*80 + "\n")
        
        # Initialize SQLite database
        try:
            conn = sqlite3.connect('sentiment_db.sqlite')
            cursor = conn.cursor()
            # Create predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_text TEXT NOT NULL,
                    predicted_sentiment VARCHAR(15) NOT NULL,
                    confidence FLOAT DEFAULT 0.0,
                    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            cursor.close()
            conn.close()
            print("SQLite database 'sentiment_db.sqlite' initialized successfully.")
        except Exception as sqlite_err:
            print(f"SQLite initialization failed: {sqlite_err}")

# Run database setup on startup
init_db()

def get_db_connection():
    """Acquires a database connection (MySQL or SQLite)."""
    if db_type == 'mysql':
        if db_pool is None:
            raise ConnectionError("MySQL database pool is not initialized.")
        return db_pool.get_connection()
    else:
        conn = sqlite3.connect('sentiment_db.sqlite')
        conn.row_factory = sqlite3.Row
        return conn

def run_query(query, params=(), fetchall=False, fetchone=False, commit=False):
    """
    Helper function to run database queries. Handles syntax differences
    between MySQL (%s) and SQLite (?) parameter placeholders automatically.
    """
    if db_type == 'sqlite':
        query = query.replace('%s', '?')
        
    conn = get_db_connection()
    
    if db_type == 'mysql':
        cursor = conn.cursor(dictionary=True)
    else:
        cursor = conn.cursor()
        
    try:
        cursor.execute(query, params)
        
        result = None
        if fetchall:
            if db_type == 'sqlite':
                rows = cursor.fetchall()
                result = []
                for r in rows:
                    d = dict(r)
                    # Convert date string to datetime object in SQLite
                    if 'prediction_time' in d and isinstance(d['prediction_time'], str):
                        try:
                            d['prediction_time'] = datetime.datetime.strptime(
                                d['prediction_time'].split('.')[0], '%Y-%m-%d %H:%M:%S'
                            )
                        except Exception:
                            pass
                    result.append(d)
            else:
                result = cursor.fetchall()
        elif fetchone:
            row = cursor.fetchone()
            if db_type == 'sqlite' and row:
                result = dict(row)
            else:
                result = row
                
        if commit:
            conn.commit()
            last_id = cursor.lastrowid
            
        cursor.close()
        conn.close()
        
        if commit and not fetchall and not fetchone:
            return last_id
        return result
    except Exception as e:
        cursor.close()
        conn.close()
        raise e

# --- PREDICTION HELPER ---
def predict_sentiment_helper(text):
    """
    Transforms the input review text using the TF-IDF vectorizer and 
    makes a sentiment classification prediction using the Logistic Regression model.
    Returns:
       predicted_class (str): 'Positive', 'Negative', or 'Neutral'
       confidence (float): The probability score of the predicted class.
    """
    if model is None or vectorizer is None:
        raise ValueError("Model and vectorizer are not loaded. Run 'python train_model.py'")
    
    # Preprocess text (lowercase and strip)
    clean_text = text.strip().lower()
    
    # Transform using vectorizer
    vectorized_text = vectorizer.transform([clean_text])
    
    # Predict class
    prediction = model.predict(vectorized_text)[0]
    
    # Calculate confidence from probabilities
    probabilities = model.predict_proba(vectorized_text)[0]
    classes = list(model.classes_)
    class_index = classes.index(prediction)
    confidence = float(probabilities[class_index])
    
    return prediction, confidence

# --- HTML WEB ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Home page handler.
    GET: Renders blank review text area.
    POST: Processes review, predicts sentiment, saves to database, and displays result.
    """
    if request.method == 'POST':
        review_text = request.form.get('review_text', '').strip()
        if not review_text:
            return render_template('index.html', error="Please enter a valid review.", db_err=db_error)
        
        try:
            # Predict
            prediction, confidence = predict_sentiment_helper(review_text)
            
            # Save to Database
            query = "INSERT INTO predictions (review_text, predicted_sentiment, confidence) VALUES (%s, %s, %s)"
            prediction_id = run_query(query, (review_text, prediction, confidence), commit=True)
            
            return render_template(
                'index.html', 
                review_text=review_text,
                prediction=prediction,
                confidence=confidence,
                error=None,
                db_err=db_error
            )
        except Exception as e:
            print(f"Error processing prediction request: {e}")
            return render_template('index.html', error=f"An error occurred: {str(e)}", db_err=db_error)
            
    return render_template('index.html', db_err=db_error)

@app.route('/history')
def history_page():
    """Fetches and displays prediction history."""
    try:
        # Fetch predictions
        query = """
            SELECT id, review_text, predicted_sentiment, confidence, prediction_time 
            FROM predictions 
            ORDER BY prediction_time DESC
        """
        predictions = run_query(query, fetchall=True)
        return render_template('history.html', predictions=predictions, db_err=db_error)
    except Exception as e:
        print(f"Error fetching history: {e}")
        return render_template('history.html', predictions=[], error=f"Failed to fetch history: {str(e)}", db_err=db_error)

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Clears all historical records in the predictions database (cascades to feedback)."""
    try:
        run_query("DELETE FROM predictions", commit=True)
    except Exception as e:
        print(f"Error clearing history: {e}")
    return redirect(url_for('history_page'))

@app.route('/admin')
def admin_page():
    """
    Aggregates database metrics and loads the admin page.
    Passes counts for Positive, Negative, Neutral, and rich analytics telemetry.
    """
    try:
        # Get count metrics
        total_predictions = run_query("SELECT COUNT(*) as count FROM predictions", fetchone=True)['count']
        pos_count = run_query("SELECT COUNT(*) as count FROM predictions WHERE predicted_sentiment = 'Positive'", fetchone=True)['count']
        neg_count = run_query("SELECT COUNT(*) as count FROM predictions WHERE predicted_sentiment = 'Negative'", fetchone=True)['count']
        neu_count = run_query("SELECT COUNT(*) as count FROM predictions WHERE predicted_sentiment = 'Neutral'", fetchone=True)['count']
        
        # Get average confidence metric
        avg_conf_row = run_query("SELECT AVG(confidence) as avg_conf FROM predictions", fetchone=True)
        avg_confidence = float(avg_conf_row['avg_conf']) if avg_conf_row and avg_conf_row['avg_conf'] is not None else 0.0
        
        # Get average character length metric
        if db_type == 'mysql':
            len_query = "SELECT AVG(CHAR_LENGTH(review_text)) as avg_len FROM predictions"
        else:
            len_query = "SELECT AVG(LENGTH(review_text)) as avg_len FROM predictions"
        avg_len_row = run_query(len_query, fetchone=True)
        avg_length = float(avg_len_row['avg_len']) if avg_len_row and avg_len_row['avg_len'] is not None else 0.0
        
        # Model telemetry
        vocab_size = len(vectorizer.vocabulary_) if vectorizer is not None else 0
        
        # Get predictions activity in the last 7 days
        if db_type == 'mysql':
            activity_query = """
                SELECT DATE_FORMAT(prediction_time, '%Y-%m-%d') as date_str, COUNT(*) as count 
                FROM predictions 
                GROUP BY date_str 
                ORDER BY date_str ASC 
                LIMIT 7
            """
        else:
            activity_query = """
                SELECT strftime('%Y-%m-%d', prediction_time) as date_str, COUNT(*) as count 
                FROM predictions 
                GROUP BY date_str 
                ORDER BY date_str ASC 
                LIMIT 7
            """
        activity_data = run_query(activity_query, fetchall=True) or []
        activity_labels = [row['date_str'] for row in activity_data]
        activity_counts = [row['count'] for row in activity_data]
        
        # Get the 5 most recent predictions
        recent_predictions = run_query(
            "SELECT id, review_text, predicted_sentiment, confidence, prediction_time FROM predictions ORDER BY prediction_time DESC LIMIT 5",
            fetchall=True
        ) or []
        
        return render_template(
            'admin.html',
            total_predictions=total_predictions,
            pos_count=pos_count,
            neg_count=neg_count,
            neu_count=neu_count,
            avg_confidence=avg_confidence,
            avg_length=avg_length,
            vocab_size=vocab_size,
            activity_labels=activity_labels,
            activity_counts=activity_counts,
            recent_predictions=recent_predictions,
            db_type=db_type,
            db_err=db_error
        )
    except Exception as e:
        print(f"Error loading admin page: {e}")
        return render_template(
            'admin.html',
            total_predictions=0, pos_count=0, neg_count=0, neu_count=0,
            avg_confidence=0.0, avg_length=0.0, vocab_size=0,
            activity_labels=[], activity_counts=[], recent_predictions=[],
            db_type=db_type,
            error=f"Database error: {str(e)}",
            db_err=db_error
        )

@app.route('/api/admin/benchmark', methods=['POST'])
def admin_benchmark():
    """
    Diagnostic benchmarking endpoint.
    Accepts JSON: { "review": "text" }
    Returns full probabilities for Positive, Negative, Neutral.
    """
    if model is None or vectorizer is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500
        
    data = request.get_json(silent=True)
    if not data or 'review' not in data:
        return jsonify({"success": False, "error": "Invalid payload"}), 400
        
    review_text = data['review'].strip()
    if not review_text:
        return jsonify({"success": False, "error": "Review text is empty"}), 400
        
    try:
        clean_text = review_text.lower()
        vectorized_text = vectorizer.transform([clean_text])
        probabilities = model.predict_proba(vectorized_text)[0]
        classes = list(model.classes_)
        
        prob_dict = {}
        for c, p in zip(classes, probabilities):
            prob_dict[c] = float(p)
            
        prediction = model.predict(vectorized_text)[0]
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "probabilities": prob_dict,
            "char_count": len(review_text),
            "word_count": len(review_text.split())
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/admin/export')
def admin_export():
    """Exports prediction database logs as a CSV file."""
    import csv
    import io
    from flask import Response
    
    try:
        predictions = run_query(
            "SELECT id, review_text, predicted_sentiment, confidence, prediction_time FROM predictions ORDER BY id DESC", 
            fetchall=True
        ) or []
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow(['ID', 'Review Text', 'Predicted Sentiment', 'Confidence', 'Prediction Time'])
        
        for pred in predictions:
            # Format time if it is a datetime object
            pred_time = pred['prediction_time']
            if hasattr(pred_time, 'strftime'):
                pred_time_str = pred_time.strftime('%Y-%m-%d %H:%M:%S')
            else:
                pred_time_str = str(pred_time)
                
            writer.writerow([
                pred['id'],
                pred['review_text'],
                pred['predicted_sentiment'],
                pred['confidence'],
                pred_time_str
            ])
            
        response = Response(output.getvalue(), mimetype='text/csv')
        response.headers["Content-Disposition"] = "attachment; filename=sentiment_predictions_export.csv"
        return response
    except Exception as e:
        return f"Error exporting predictions: {str(e)}", 500

# --- REST API ENDPOINTS ---

@app.route('/predict', methods=['POST'])
def predict_api():
    """
    REST API endpoint for predicting sentiment.
    Expects JSON: { "review": "Text to analyze" }
    Returns JSON: { "success": true, "sentiment": "Positive/Negative/Neutral", "confidence": 0.92 }
    """
    data = request.get_json(silent=True)
    if not data or 'review' not in data:
        return jsonify({
            "success": False,
            "error": "Invalid request parameters. Please supply JSON with a 'review' field."
        }), 400
        
    review_text = data['review'].strip()
    if not review_text:
        return jsonify({
            "success": False,
            "error": "Review field is empty."
        }), 400

    try:
        # Perform prediction
        prediction, confidence = predict_sentiment_helper(review_text)
        
        # Save prediction to DB
        query = "INSERT INTO predictions (review_text, predicted_sentiment, confidence) VALUES (%s, %s, %s)"
        prediction_id = run_query(query, (review_text, prediction, confidence), commit=True)
        
        return jsonify({
            "success": True,
            "sentiment": prediction,
            "confidence": confidence
        })
    except Exception as e:
        print(f"API Prediction Error: {e}")
        return jsonify({
            "success": False,
            "error": f"Failed to predict: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Start the Flask development server
    app.run(host='127.0.0.1', port=5000, debug=True)
