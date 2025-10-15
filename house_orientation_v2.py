import cv2
import numpy as np
import pytesseract
import math

def bearing_from_vec(dx, dy):
    ang = math.degrees(math.atan2(dx, -dy)) % 360
    return ang

def compass8(b):
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    return dirs[int((b + 22.5)//45)%8]

def get_text_boxes(image):
    """Run OCR and return (text, (cx, cy)) for each detected word."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)[1]

    data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    results = []
    for i, text in enumerate(data["text"]):
        if text.strip():
            (x, y, w, h) = (data["left"][i], data["top"][i],
                            data["width"][i], data["height"][i])
            cx, cy = x + w/2, y + h/2
            results.append((text.strip(), (cx, cy)))
    return results

def find_orientation(img_path, target_label):
    img = cv2.imread(img_path)
    texts = get_text_boxes(img)

    # locate the house label (e.g., "13")
    house_points = [p for t, p in texts if t == str(target_label)]
    if not house_points:
        raise ValueError(f"Could not find label {target_label}")
    house = np.mean(house_points, axis=0)

    # locate road labels (heuristic: long words, likely top of image)
    road_candidates = [(t, p) for t, p in texts if len(t) > 4]
    if not road_candidates:
        raise ValueError("No road label detected")
    # pick nearest road
    road, road_point = min(road_candidates,
                           key=lambda tp: np.linalg.norm(np.array(tp[1]) - house))

    # compute vector & bearing
    dx, dy = road_point[0] - house[0], road_point[1] - house[1]
    b = bearing_from_vec(dx, dy)
    dir8 = compass8(b)

    # visualize
    vis = img.copy()
    cv2.arrowedLine(vis, tuple(np.int32(house)),
                    tuple(np.int32(road_point)), (0,255,255), 2, tipLength=0.1)
    cv2.putText(vis, f"{dir8} ({b:.1f}°)", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Result", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"House {target_label} faces {dir8} ({b:.1f}°) toward {road}")

if __name__ == "__main__":
    find_orientation("houses.png", target_label=13)
