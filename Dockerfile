# Hugging Face Spaces (Plano A) e Render (Plano B): roda o "Terminal de triagem" (server.py + web/).
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Pré-calcula IF e PELT (80 combinações) na construção da imagem: o app abre rápido na nuvem.
RUN python -c "import server; server.Estado()"
# Hugging Face usa a porta 7860; o Render define a variável PORT.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "python server.py --host 0.0.0.0 --port ${PORT}"]
