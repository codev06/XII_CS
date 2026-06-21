from flask import Flask, render_template, request, jsonify

# Connect independent functional modules
from sql_fetcher import MySQLDataFetcher
from textblob_predictor import TextBlobPredictor
from sgd_tracker import ModuleAlphaSGD
from Accuracy_optimizer import AccuracyOptimizer

user_text = ""  # Global variable to store user input text
app = Flask(__name__)

# SYSTEM DATABASE HOST CONFIGURATION
# Adjust credentials matching your specific target instance properties
db_fetcher = MySQLDataFetcher(
    host="localhost",
    user="root",
    password="seanc0de!",  # <-- Update with your actual password
    database="sentiment_db",
    port=3306
)

# Initialize standalone analytical modules
predictor = TextBlobPredictor(csv_path="polarity_thresholds.csv")
trend_tracker = ModuleAlphaSGD()
optimizer = AccuracyOptimizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/fetch_db', methods=['POST'])
def fetch_from_db():
    record_id = request.form.get('record_id', '').strip()
    if not record_id:
        return jsonify({"error": "Missing unique identifier input key value"}), 400
    try:
        extracted_text = db_fetcher.fetch_record(
            table="conversations", 
            column="raw_payloadtext", 
            record_id=int(record_id)
        )
        if not extracted_text:
            return jsonify({"error": "No data found matching input parameter row index key"}), 404
        return jsonify({"text": extracted_text})
    except ValueError:
        return jsonify({"error": "Primary identification key values must be valid integers"}), 400

@app.route('/analyze', methods=['POST'])
def analyze_stream():
    global user_text
    user_text = request.form.get('text', '').strip()
    if not user_text:
        return jsonify({"error": "Payload processing validation exception: Text is empty"}), 400
        
    # Cascade metrics down across sequential operations processing loops
    tb_analysis = predictor.analyze(user_text)
    predicted_next = trend_tracker.update_and_predict(tb_analysis['score'])
    fused_results = optimizer.fuse_and_compare(tb_analysis, predicted_next)
    
    return jsonify({
        "raw_text": user_text,
        "current_metrics": tb_analysis,
        "predicted_next_trend": round(predicted_next, 4),
        "fused_metrics": fused_results
    })

if __name__ == '__main__':
    print("Application successfully built from source files. Booting environment server...")
    app.run(debug=True, port=5000)