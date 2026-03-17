import requests
import json

# 配置 DeepSeek API
API_KEY = "在这里填入你的 DeepSeek API Key"
BASE_URL = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def chat_with_deepseek(user_input):
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": user_input}]
    }
    response = requests.post(BASE_URL, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"请求失败: {response.status_code}, {response.text}"

if __name__ == "__main__":
    print("DeepSeek Chatbot 已启动，输入 'quit' 退出。")
    while True:
        user_input = input("你: ")
        if user_input.lower() == "quit":
            print("再见！")
            break
        reply = chat_with_deepseek(user_input)
        print(f"DeepSeek: {reply}")
