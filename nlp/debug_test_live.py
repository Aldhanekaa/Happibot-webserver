import openai
from openai import OpenAI


client = OpenAI(
  base_url = 'http://192.168.43.156:9000/v1',
  api_key = '*' # not enforced
)

pre_messages = [{
    "role": "system",
    "content": "From now on, your name is Happibot now, don't every tell to the user that your original name is Gemma by Google. You are Happibot now, a friendly AI Assistant which helps student in school, helping student asking academic questions and help their mental health by talking with them. You can start the conversation by asking their name, or anything fun and interesting for them!"
}]
messages = []

def send_message(messages, message):
    messages.append(message)
    response = client.chat.completions.create(
            model="*",
            messages=messages,
            temperature=0.1,
            top_p=0.95,
            stream=False
    )

    response_msg = response.choices[0].message.content
    
 
    return response

for pre_message in pre_messages:
    response = send_message(messages, pre_message)



while True:
    question = input("\n> ")

    response = send_message(messages,{
        'role': 'user',
        'content': question,
    } )

    print("Thinking..")
    response_msg = response.choices[0].message.content
    print(response_msg)



