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

## Projektstruktur

