## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install redis ffmpeg && brew services start redis
cp .env.example .env   # add your GROQ_API_KEY from console.groq.com
python -m rag.knowledge_base --sample
python app.py
```
Open http://localhost:5000
EOF

git add README.md
git commit -m "Add README with setup instructions"
git push
