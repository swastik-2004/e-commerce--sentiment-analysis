🧠 Sentiment Analysis using DistilBERT

An end-to-end Sentiment Analysis System that classifies product reviews as Positive, Neutral, or Negative using DistilBERT from Hugging Face Transformers.

Built with:

🧩 Hugging Face Transformers (for fine-tuning)

⚡ PyTorch (for GPU acceleration)

🌐 Streamlit (for real-time inference)

📊 Matplotlib / Seaborn (for model evaluation)

🚀 Features

✅ Fine-tuned DistilBERT model on 500K+ cleaned Amazon product reviews
✅ Interactive Streamlit web app for live sentiment prediction
✅ Real-time confidence score visualization and feedback
✅ Automatic prediction logging for analytics
✅ Modular, production-ready code structure



🧠 Model Overview

Base Model: distilbert-base-uncased
Fine-tuned on: Amazon product reviews (cleaned & labeled)
Classes:

0 → Negative

1 → Neutral

2 → Positive

Final Evaluation:

Metric	Score
Accuracy	88.8%
Weighted F1	0.89
Dataset Size	~500,000 reviews
Model Size	~250MB
⚙️ Installation
1️⃣ Clone this repo
git clone https://github.com/<swastik-2004>/sentiment_analysis_project.git
cd sentiment_analysis_project

2️⃣ Create and activate environment
conda create -n torch_gpu python=3.10
conda activate torch_gpu

3️⃣ Install dependencies
pip install -r requirements.txt

🧩 Usage
▶️ Run the Streamlit app
streamlit run app.py


Open your browser at http://localhost:8501

Example input:

“The product was very mediocre. It had flaws but still worked.”

Example output:

Prediction: Neutral
Confidence: 96.9%

📊 Dashboard (Optional Add-On)

A secondary Streamlit page that visualizes:

Sentiment distribution (Pie Chart)

Daily sentiment trends (Line Chart)

Recent predictions table

📁 Model Training Summary
Steps:

Cleaned dataset (Cleaned_Review, Sentiment)

Mapped labels → 0, 1, 2

Tokenized with AutoTokenizer

Fine-tuned DistilBERT with Trainer

Saved final model to model/sentiment_bert/

Training Time: ~70s per epoch on RTX 3050
Loss: 0.43 after epoch 1
Eval Accuracy: ~88%

🧠 Tech Stack
Area	Tools Used
Data Cleaning	Pandas, Regex
NLP Model	Hugging Face Transformers (DistilBERT)
Training Framework	PyTorch, Trainer API
Evaluation	Sklearn (F1, Accuracy, Confusion Matrix)
UI	Streamlit
Visualization	Matplotlib, Seaborn
Logging	CSV-based tracking for predictions
🧾 Example Resume Line

Developed an end-to-end Sentiment Analysis Web App using DistilBERT, achieving 89% accuracy on 500K product reviews. Built a Streamlit UI with real-time confidence visualization and analytics logging.

🌍 Future Improvements

✅ Add FastAPI backend for scalable deployment

✅ Integrate SQLAlchemy for user-based storage

✅ Host model and app on Hugging Face Spaces or Streamlit Cloud

✅ Add Admin Dashboard for trend monitoring

🧑‍💻 Author

Swastik Dasgupta
🎓 3rd Year AIML, MSRIT
💼 Aspiring Machine Learning Engineer

⭐ Acknowledgements

Hugging Face Transformers

Streamlit

PyTorch