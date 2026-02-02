# GLiNER PDF Service

Dieses Projekt stellt einen **Docker-basierten Webservice** bereit, mit dem
PDF-Dokumente getestet werden können, indem:

- Text aus PDFs mit **Docling + RapidOCR** extrahiert wird
- trainierte **GLiNER NER-Modelle** auf den Text angewendet werden
- erkannte Entitäten und Evidenzen angezeigt und heruntergeladen werden

⚠️ **Trainierte Modelle sind bewusst NICHT Teil dieses Repositories**
und müssen separat bereitgestellt werden.

---

## Voraussetzungen

Für die Nutzung werden benötigt:

- **Docker** (Docker Desktop empfohlen)
- **Git**
- separat bereitgestellte **GLiNER-Modelle**

Eine lokale Python-Installation ist **nicht erforderlich**, wenn Docker genutzt wird.

---

## Anleitung

### 1. Repository herunterladen

git clone https://github.com/laurawagner-ger/gliner-ubo-ocr-service.git
cd gliner-ubo-ocr-service


### 2. Modelle lokal hinzufügen

Lege die trainierten GLiNER-Modelle im Ordner models/ ab.

Beispiel: ubo_gliner_base, ubo_gliner_small


 ### 3. Docker Image bauen
docker build -t gliner-pdf-service .


### 4. Service starten
docker run -p 8501:8501 gliner-pdf-service


### 5. Weboberfläche öffnen

Im Browser aufrufen:

http://localhost:8501

Der Service ist nun einsatzbereit.



### Service stoppen
Ctrl + C
