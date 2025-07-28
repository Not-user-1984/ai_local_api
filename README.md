1. Скачай модели и положи в ./models/


```bash
mkdir -p models
cd models

# Универсальная
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf -O mistral.gguf

# Кодовая
wget https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf -O codellama.gguf
```


2. Собери и запусти контейнер

```bash
docker-compose up --build -d
```

3. Проверь

```bash
curl -X POST http://localhost:8000/chat/general -d "prompt=Привет, кто ты?"
```
