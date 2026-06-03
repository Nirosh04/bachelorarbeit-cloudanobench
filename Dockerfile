FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY notebooks/ notebooks/
COPY data/ data/
COPY docs/ docs/

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--LabApp.token=''"]