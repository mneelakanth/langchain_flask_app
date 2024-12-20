from flask import Flask, jsonify, request, render_template
import os
import argparse

from utils import split_text, process_file, create_embeddings, create_pipeline, create_vector_db

app = Flask(__name__)

## Set the file upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

### Allowed extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

def allowed_file(filename):
    """Check if the file has allowed extensions."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def arg_parse():
    parser = argparse.ArgumentParser(description="Run the Flask web app")
    parser.add_argument('--modelname', type=str, default='huggingface', help="Model name for llm.")
    return parser.parse_args()
args = arg_parse()

embeddings = create_embeddings(args.modelname)
db = None
llm = create_pipeline(args.modelname)

def process_data(content):
    docs = split_text(content)
    db = create_vector_db(docs, embeddings)
    return db

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Process the uploaded file
            file_type = filepath.split('.')[-1]
            content = process_file(filepath, file_type)
            
            return render_template('index.html', content=content, filename=filename)
        else:
            return 'Invalid file type. Only .txt and .csv files are allowed.', 400
    return render_template('index.html', content=None)

@app.route('/remove', methods=['GET'])
def remove_content():
    content = None
    return render_template('index.html', content=None)

@app.route('/ask_question', methods=['POST'])
def ask_question():
    """Handle questions based on the file content."""
    question = request.form['question']
    content = request.form['content']
    if args.modelname == 'huggingface':
        result = llm(question=question, context=content)

        result = result['answer']
        return jsonify({"answer": result})

    if args.modelname == 'genai':
        db = process_data(content)
        relevant_docs = db.similarity_search(question, k=3)
        context = " ".join([doc.page_content for doc in relevant_docs])

        prompt = f"Question: {question}\nContext: {context}\nAnswer:"

        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        answer = llm.invoke(messages)
        answer = answer.content
        return jsonify({'answer': answer})

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
