from quart import Quart, request, jsonify
import torch
from train import SimpleRNN, TEXT

app = Quart(__name__)

# モデルのロード
model = SimpleRNN(len(TEXT.vocab), 256, len(TEXT.vocab))
model.load_state_dict(torch.load('./model/trained_model.pth'))
model.eval()

@app.route('/chat', methods=['POST'])
async def chat():
    data = await request.get_json()
    input_text = data['text']
    
    # 入力テキストのトークン化
    tokens = TEXT.preprocess(input_text)
    token_indices = [TEXT.vocab.stoi[token] for token in tokens]
    input_tensor = torch.LongTensor(token_indices).unsqueeze(0)
    
    # モデルによる予測
    with torch.no_grad():
        output = model(input_tensor)
    predicted_token = TEXT.vocab.itos[output.argmax(dim=1).item()]
    
    return jsonify({'response': predicted_token})

if __name__ == '__main__':
    app.run()
