# Gangerkennung mit OpenPose und Machine Learning

Es soll in diesem Projekt versucht werden mit Hilfe der 2D-Gelenkpositionbestimmung des *OpenPose* Frameworks Menschen anhand ihres Ganges zu klassifizieren. Der analysierte Datensatz wurden hierzu eigenständig erstellt und besteht aus 20 verschiedenen Probanden von denen jeweils 30 Gänge aufgenommen wurden.

---
> **_Notiz:_** Falls ihr auf den Datensatz für weiterführende Projekte zugreifen wollt, meldet Euch bitte bei dem Owner dieses Repositories. Die Gänge wurden aus zwei verschiedenen Perspektiven mit unterschiedlichen Kameras aufgenommen. Die Aufnahme der sagitalen Ebene erfolgte durch eine Kamera (50 fps). Zusätzlich wurde die fronatal Ebene mit einer Tiefenkamera aufgenommen. Die Synchronisation beider Kameras erfolgte über ein Licht, das vor Beginn des Ganges ein- und ausgeschaltet wurde. Für die bisherige Analyse der Gänge wurde bisher nur die Videos aus der sagitalen Ebene genutzt.
---
 
## Analysieren von Videos mit OpenPose

Die Videos der Gänge wurden mit OpenPose analysiert. Es wird hierzu die *2D-Pose-Estimation* von OpenPose genutzt, die [25 verschiedene Gelenkpunkte](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md#pose-output-format-body_25 "Übersicht zu den Gelenkpunkten") in den Videos erkennen kann. Die Verarbeitung der Videos kann auf zwei verschiedene Arten erfolgen:
* [Lokale Installation](https://github.com/CMU-Perceptual-Computing-Lab/openpose#installation "Anleitung zur Installation von OpenPose") von OpenPose auf einem Computer. Für schnelle Verarbeitung der Videos empfiehlt es sich die GPU Version zu installieren. Es wird ebenso ein Computer benötigt der ausreichend GPU zur Verfügung steht. 
* Eine weitere Möglichkeit ist die Verarbeitung der Videos online mit Hilfe des zur Verfügung gestellten Jupyter-Notebooks `google-colab-openpose-notebook.ipynb`. Hierbei wird OpenPose auf einer von Google kostenlos zur Verfügung gestellten Landschaft installiert und die Videos verarbeitet. Die genaue Vorgehensweise und notwendigen Schritte werden in dem bereitgestellten Notebook beschrieben.

## Lokales Setup

Für die Verarbeitung der JSON-Daten und die anschließende Merkmalextraktion und Klassifizierung werden verschiedene Python-Bibliotheken benötigt. Diese stellen bspw. Klassifzierungsalgorithmen bereit oder helfen beim Einlesen und Ausschreiben von Daten in verschiedene Formate. Aktuell werden folgende Python-Bibliotheken in diesem Projekt genutzt:
* [Pandas](https://pandas.pydata.org/pandas-docs/stable/index.html)
* [Numpy](https://numpy.org/)
* [Scikit-Learn](https://scikit-learn.org/stable/)
* [Matplotlib](https://matplotlib.org/)

Für die Installation dieser Packages führe folgendes Kommando in deiner Python-Umbegung aus:
```
pip install -r requirements.txt
```
Nach der Ausführung dieses Kommandos sollten alle benötigten Programmbibliotheken auf deinem Computer installiert sein. Falls dir bei der Ausführung des Programmcodes Fehler auftreten, weil ein Package nicht installiert wurde, füge dies bitte dem Readme und der Datei `requirements.txt` hinzu, damit dieses Problem bei anderen nicht mehr auftritt. 

---
> **_Notiz:_**: Falls neue Funktionalitäten, die weitere Python-Bibliotheken benötigten, hinzugefügt werden, updatet bitte sowohl die Liste der genutzten Programmbibliotheken als auch die Datei `requirements.txt`. Dies erleichert die lokale Einrichtung und Installtion für die Software und spart daher einiges an Zeit.
---

Für eine möglichst einfache Anwendung der bereitgestellten Programme empfiehlt es sich folgende Punkte umzusezten:
* Die Bennung der Videos sollte im folgendem Format erfolgen: Jahr_Monat_Tag_Proband_Session. Der Proband kann hierbei durch einen Buchstaben abgekürzt werden. Die Session beginnt bei jedem Video bei 1 und steigt inkrementell an. Gleichzeitig sollte darauf geachtet werden, dass bei der Verarbeitung der Videos mit OpenPose die Keypoints in einem Ordner mit demselbem Format abgespeichert werden. Hierdurch lassen die Keypoints ohne Probleme einem Video zuordnen
* Falls die Bennennung der Ordner nicht wie in der folgendem Abbildung dargestellt sind, empfiehlt es sich diese anzupassen. Ansonsten sind für die Nutzung der Software weitere Anpassung notwendig.

![Ordnerstruktur](/docs/images/ordnerstruktur.jpg)

* Die korrekte Bennung der Unterordner, die die JSON-Daten ist sehr wichtig, da diese für die Bennung der erzeugten CSV-Dateien genutzt werden. Anhand des Names der CSV-Dateien wird außerdem das Label (die Person) für den Training- und Testdatensatz bestimmt. Deswegen sollten spätestens zu diesem Zeitpunkt eine korrekte Bennung der CSV-Dateien vorliegen. Ansonsten kann es dazu kommen, dass die Daten nicht korrekt gelabeld werden.

![CSV-Dateien](/docs/images/csv.jpg)

Die empfohlene Bennung und Strutkuierung der Ordner und Dateien ist auf den jeweiligen Bilder zur Veranschaulichung dargestellt.

## Überblick der bereitgestellten Komponenten

**data_reader.py**: Diese Komponente kann zum Einlesen der Gelenkpunkte aus den erzeugten JSON-Dateien genutzt werden. Es werden hierbei die Punkte *pose_keypoints_2d* aus jeder Datei eingelesen und Dateien, die im gleichem Ordner abgelegt wurden, zu einer Session zusammengefasst. Des Weiteren können durch diese Komponten CSV-Dateien, die Gangmerkmale (Winkel, Kadenz, Pixel Distanz) enthalten, eingelesen werden. Es werden alle CSV-Dateien, die sich im selben Ordner befinden eingelesen. Zusätzlich wird anhand des Dateiname, die aufgenommene Person bestimmt. Es ist deshalb besonders wichtig sich an die korrekte Bennung zu halten.

**gait_utils.py**: Diese Komponente stellt verschiedene Funktionalitäten bereit, um die Gänge auszuwerten bzw. die Gangmerkmale zu bestimmen. Ein Großteil der Implementierung orientierte sich an dem [LAPSHub Repo](https://github.com/LAPSHub/gait-joint-angles), die bereits Methode zur Analyse der OpenPose Daten bereitstellen. Trotzdem wurden bei veerschiedenen Methoden Anpassungen durchgeführt, um bessere Ergebnisse zu erzielen. Insbesondere bei der Erkennung von Gangevents besteht aber weiterhin viel Verbesserungspotential. 

**extract_features.py**: In dieser Komponente werden anhand der zuvor eingelesenen Gelenkpunkte verschiedene Gangermerkmale bestimmt und Gangevents ermittelt. Hierzu werden die Winkel mit den Funktionen aus der Komponente *gait_utils.py* berechnet und nach der Bestimmung der initalen Bodenkontakte auf diesen Zeitraum gekürzt. Des Weiteren werden Merkmale, wie z.B. die Kadenz und die Pixel Distanz eines Doppelschrittes anhand der initalen Bodenkontakte ermittelt. Abschließend werdend alle diese Merkmale in eine CSV-Datei geschrieben. Diese hat denselben Namen, wie der Ordner der JSON-Dateien, die für die Bestimmung der Merkmale gentutz wurden. Wenn kein Doppelschritt anhand der Daten der Videos erkannt werden konnte, wird eine Fehlermeldung ausgegeben und keine CSV-Datei für dieses Video erstellt.

**data_normalization.py**: Diese Komponente wird zur Normalisierung der Gangmerkmale genutzt. Dies ist ein notwendiger Schritt, damit mit diesen Daten eine Klassifzierung durchgeführt werden kann. Es werden hierzu die verschiedenen Gangmerkmale aus den zuvor erstellten CSV-Dateien ausgelesen und anschließen durch eine lineare Interpolation auf eine einheitliche Länge gebracht. Außerdem besteht die Möglichkeit Ungnauigkeiten der Winkel durch die Nutzung eines Gauß-Filters auszugleichen. Dieser sowohl vor als auch nach der linearen Interpolation angewendet werden. Des Weiteren besteht eine Auswahl der Merkmale für das Training durchzuführen. Zusätzlich wird eine Methode zur Formatierung der Merkmale bereitgestellt, sodass diese für das Training der statistischen Klassifizierer genutzt werden können. Dies geschieht durch eine Umformung (reshape) des vorliegenden Numpy-Arrays 

**classifier.py**: In dieser Komponente sind die verschiedene Funktionalitäten, die für die Klassifzierung der Daten notwendig sind, implementiert. Es wird eine Methode für die Aufteilung der Daten in Trainings- und Testdaten bereitgestellt. Außerdem können zur Zeit fünf verschiedene statistische Klassifizierer trainiert werden. Eine Beurteilung der Klassifizierer durch den *F1-Score* ist ebenfalls in dieser Komponente implementiert.  

**plot_data.py**: Diese Komponenten kann zur Darstellung der Winkelverläufe genutzt werden. Dies ist insbesondere sinnvoll, um fehlerhafte Ausgaben nachzuvollziehen und Probleme (z.B. Ermittlung von Gangevents) zu visualieren.

**test_script.py**: Dieses Skript soll die Nutzung der verschiedenen Komponenten durch eine konkrete beispielhafte Implementierung vereinfachen.

**google-colab-openpose-notebook.ipyn**: Mit diesem Notebook kann, wie bereits zuvor beschrieben, OpenPose auf Google Colab installiert und für die Analyse der Videos genutzt werden. Für die Nutzung muss dieses Notebook einfach auf Colab hochgeladen und anschließend den beschriebenden Schritte gefolgt werden. 

## Ausblick

Folgende Tätigkeiten wurden entweder bereits abgeschlossen oder es wird empfohlen diese in den nächsten Schritten zu implementieren, um bessere Ergebnissse mit dem Prototypen zu erzielen bzw. den Prototypen um weitere Funktionalitäten zu erweitern.

- [x] Google Colab Notebook für die Nutzung von OpenPose
- [x] Einlesen und Verarbeitung der JSON-Dateien
- [x] Erkennung von Gangevents (initalen Bodenkontakt)
- [x] Extraktion von Gangmerkmale (beliebige Winkel, Kadenz, Pixel Distanz)
- [x] Normalisierung der Gangwinkel mittels linearer Interpolation
- [x] Aufteilungen der Daten in Trainings- und Testdaten
- [x] Implementierung statistischer Klassifizierer (KNN, SVM, Naive Bayes, Random Forests)
- [x] Metriken zur Evaluierung der Klassifizierungsergebnisse
- [ ] Diagramme erstellen für die berechneten Metriken

- [ ] Bessere Fehlerbehandlung
- [ ] Neuronales Netz zur Klassifzierung der Gangdaten nutzen
- [ ] Klassifizierung mittels 3D-Open-Pose Daten aus der frontalen Ebene
- [ ] Vergleich der Winkel zwischen OpenPose 2D und 3D Pose Estimation
- [ ] Erkennung weiterer Gangevents und zusätzliche Wege zur Erkennung des initalen Bodenkontaktes
- [ ] Genauere Analyse der Videos auf welchen kein Doppelschritt erkannt werden konnte
- [ ] Dynamic Time Warping zum Vergleich der Winkelverläufe
