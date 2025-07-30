from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# Load the trained model and tokenizer
model_path = "./chatbot_model"  # Path where the model and tokenizer are saved
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        # Get input data from the request
        data = request.get_json()
        if 'message' not in data:
            return jsonify({"error": "Please provide a 'message' in the request body"}), 400
        
        user_message = data['message']
        
        # Encode the input message
        input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt').to(device)
        
        # Generate a response
        output_ids = model.generate(input_ids, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id)
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Return the response
        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
