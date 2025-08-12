from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/v1/engines/davinci-codex/completions', methods=['POST'])
def completions():
    # 获取请求中的JSON数据
    data = request.json
    
    # 模拟生成的完成文本
    response_text = "This is a generated text based on the input prompt."
    
    # 构建响应对象
    response = {
        "id": "cmpl-6HJZGKQZJHJZGKQZJHJZGKQZ",
        "object": "text_completion",
        "created": 1597028234,
        "model": "davinci-codex",
        "choices": [
            {
                "text": response_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "length"
            }
        ],
        "usage": {
            "prompt_tokens": len(data["prompt"].split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(data["prompt"].split()) + len(response_text.split())
        }
    }

    

    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
