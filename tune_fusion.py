import pandas as pd
from fusion_layer import fuse_detections
# Importiere hier die Funktion zum Laden der Test-Daten
# from data_loader import load_data 

def run_tuning(event_boxes, rgb_boxes):
    # Definition der Bereiche, die wir testen wollen
    event_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    rgb_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    iou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    results = []

    print("Starte Grid-Search Tuning...")

    for et in event_thresholds:
        for rt in rgb_thresholds:
            for it in iou_thresholds:
                # Hier fusionieren wir mit den aktuellen Parametern
                fused_data = fuse_detections(event_boxes, rgb_boxes, event_t=et, rgb_t=rt, iou_t=it)
                
                # Hier muss jetzt die mAP-Berechnungs-Funktion aufgerufen werden
                # score = calculate_map(fused_data) 
                score = 0.0 # Platzhalter für das Ergebnis der mAP-Berechnung
                
                results.append({
                    'event_t': et, 
                    'rgb_t': rt, 
                    'iou_t': it, 
                    'map50': score
                })
                print(f"Getestet: E={et}, R={rt}, IOU={it} -> mAP50={score}")

    # Ergebnisse sortieren und das Beste anzeigen
    df = pd.DataFrame(results)
    best_result = df.sort_values(by='map50', ascending=False).iloc[0]
    
    print("\nBeste Konfiguration gefunden:")
    print(best_result)
    return df

# Startpunkt: Hier unsere Daten laden und Tuning starten
# event_data, rgb_data = load_data('path/to/your/data')
# df_results = run_tuning(event_data, rgb_data)