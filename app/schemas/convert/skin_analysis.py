def acne_detection_to_response(analysis):
    acneDetection = {
        "meta": {
            "classes": {
                "0": "acne_scars",
                "1": "blackhead",
                "2": "cystic",
                "3": "flat_wart",
                "4": "folliculitis",
                "5": "keloid",
                "6": "milium",
                "7": "papular",
                "8": "purulent",
                "9": "sebo-crystan-conglo",
                "10": "syringoma",
                "11": "whitehead"
            },
            "conf_threshold": 0.5
        },
        "predicts": []
    }

    acneSeverity = {
        "meta": {
            "classes": getattr(analysis.acne_severity, "available_classes", {}),
            "conf_threshold": 0.5
        },
        "predicts": [
            {
                "name": getattr(analysis.acne_severity, "severity_name", None),
                "confidence": getattr(analysis.acne_severity, "confidence", None),
                "classes": getattr(analysis.acne_severity, "severity_class", None),
            }
        ]
    }

    skinType = {
        "meta": {
            "classes": {
                "0": "Combination",
                "1": "Dry",
                "2": "Normal",
                "3": "Oily"
            }
        },
        "predicts": {
            "name": getattr(analysis.skin_type, "skin_type_name", None),
            "confidence": getattr(analysis.skin_type, "confidence", None),
            "class_index": getattr(analysis.skin_type, "class_index", None)
        }
    }

    for acne in getattr(analysis, "acne_detections", []):
        box = {
            "x1": acne.x1,
            "y1": acne.y1,
            "x2": acne.x2,
            "y2": acne.y2,
        }
        color = [acne.color_r, acne.color_g, acne.color_b]
        item = {
            "name": acne.name,
            "class": acne.class_id,
            "confidence": acne.confidence,
            "box": box,
            "color": color,
        }
        acneDetection["predicts"].append(item)

    return {
        "acneDetection": acneDetection,
        "acneSeverity": acneSeverity,
        "skinType": skinType,
        "image_url": getattr(analysis, "image_url", None),
        "id": getattr(analysis, "id", None),
        "user_id": getattr(analysis, "user_id", None),
        "created_at": getattr(analysis, "created_at", None),
        "updated_at": getattr(analysis, "updated_at", None),
    }
