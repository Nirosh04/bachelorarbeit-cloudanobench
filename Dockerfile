FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY notebooks/ notebooks/
COPY tests/ tests/
COPY docs/ docs/

# data/ wird nicht ins Image kopiert (Rohdaten nicht im Repo).
# Beim Start per Volume mounten: -v "${PWD}/data:/app/data"
RUN mkdir -p data

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--LabApp.token=''"]
