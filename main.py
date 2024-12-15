from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize the Flask app
app = Flask(__name__)

# Load the BLOOM model and tokenizer
model_name = "bigscience/bloom-560m"  
print("Loading the model... This may take a while.")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Model loaded successfully!")

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat requests.
    """
    try:
        user_input = request.json.get('message', '')
        if not user_input:
            return jsonify({"error": "No input message provided"}), 400

        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt")

        # Generate a response
        outputs = model.generate(
            inputs['input_ids'], 
            max_length=150, 
            num_return_sequences=1, 
            temperature=0.7,  # Adjust for randomness
            top_p=0.9  # Adjust for diversity
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  
    app.run(host='0.0.0.0', port=port)
