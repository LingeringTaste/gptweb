from flask import Flask, request, jsonify, render_template, Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import json
import uuid

load_dotenv()

app = Flask(__name__)

# Database configuration
if os.environ.get('WEBSITE_HOSTNAME'):  # Running on Azure
    # Azure Database for PostgreSQL connection string
    db_user = os.environ.get('POSTGRES_USER')
    db_password = os.environ.get('POSTGRES_PASSWORD')
    db_host = os.environ.get('POSTGRES_HOST')
    db_name = os.environ.get('POSTGRES_DB')
    
    # PostgreSQL connection string
    app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{db_user}:{db_password}@{db_host}/{db_name}'
else:  # Local development
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///conversations.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    conversation_id = db.Column(db.String(50), nullable=False)  # to group messages in same conversation
    username = db.Column(db.String(50), nullable=False)  # to identify the user
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    try:
        username = request.json.get('username')
        if not username:
            return jsonify({'error': 'Username is required'}), 400
            
        db.session.query(Message).filter_by(username=username).delete()
        db.session.commit()
        return jsonify({'status': 'success'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        api_key = data.get('api_key', '')
        system_prompt = data.get('system_prompt', '')
        config = data.get('config', {})
        conversation_id = data.get('conversation_id', str(uuid.uuid4()))
        username = data.get('username', '')

        if not message or not api_key or not username:
            return jsonify({'error': 'Message, API key, and username are required'}), 400

        # Initialize messages with system prompt if provided
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Get only the last few messages for context (e.g., last 10 messages)
        conversation_messages = Message.query.filter_by(
            conversation_id=conversation_id, 
            username=username
        ).order_by(Message.timestamp.desc()).limit(10).all()
        
        # Reverse to get chronological order
        conversation_messages.reverse()
        
        # Add context messages
        for msg in conversation_messages:
            messages.append({"role": msg.role, "content": msg.content})

        # Add current message
        messages.append({"role": "user", "content": message})

        # Create timestamp for both messages
        user_timestamp = datetime.now()
        
        # Save user message to database
        user_msg = Message(
            content=message,
            role='user',
            conversation_id=conversation_id,
            username=username,
            timestamp=user_timestamp
        )
        db.session.add(user_msg)
        db.session.commit()

        def generate():
            nonlocal user_msg  # Add this to access the outer variable
            
            # Call Azure OpenAI API
            client = AzureOpenAI(
                azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
                api_key=api_key
            )

            full_response = ""
            assistant_timestamp = user_timestamp + timedelta(milliseconds=100)

            try:
                with app.app_context():  # Add application context here
                    response = client.chat.completions.create(
                        model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
                        messages=messages,
                        temperature=config.get('temperature', 0),
                        max_tokens=config.get('max_tokens', 1000),
                        top_p=config.get('top_p', 1),
                        frequency_penalty=config.get('frequency_penalty', 0),
                        presence_penalty=config.get('presence_penalty', 0),
                        stream=True
                    )

                    for chunk in response:
                        if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                full_response += content
                                yield f"data: {json.dumps({'content': content, 'done': False})}\n\n"

                    # Save the complete response to database
                    assistant_msg = Message(
                        content=full_response,
                        role='assistant',
                        conversation_id=conversation_id,
                        username=username,
                        timestamp=assistant_timestamp
                    )
                    db.session.add(assistant_msg)
                    db.session.commit()

                    # Send the final message with done flag
                    yield f"data: {json.dumps({'content': '', 'done': True, 'timestamp': assistant_timestamp.isoformat(), 'conversation_id': conversation_id})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    username = request.args.get('username')
    if not username:
        return jsonify({'error': 'Username is required'}), 400

    try:
        # Get all messages for this user, ordered by timestamp
        messages = Message.query.filter_by(
            username=username
        ).order_by(Message.timestamp).all()
        
        return jsonify({
            'messages': [{
                'content': msg.content,
                'role': msg.role,
                'timestamp': msg.timestamp.isoformat(),
                'conversation_id': msg.conversation_id
            } for msg in messages]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
