import ast
import cv2
import numpy as np
import pandas as pd


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)

    return img


def visualize_results(input_csv, video_path, output_video_path):
    results = pd.read_csv(input_csv)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    license_plate = {}
    for car_id in np.unique(results['car_id']):
        max_score = np.amax(results[results['car_id'] == car_id]['license_plate_bbox_score'])
        license_plate_data = results[(results['car_id'] == car_id) & (results['license_plate_bbox_score'] == max_score)]
        if not license_plate_data.empty:
            license_plate[car_id] = license_plate_data.iloc[0]

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_data = results[results['frame_nmr'] == frame_number]
        for _, row in frame_data.iterrows():
            try:
                car_bbox = ast.literal_eval(row['car_bbox'])
                license_plate_bbox = ast.literal_eval(row['license_plate_bbox'])

                if row['car_id'] in license_plate:
                    license_number = license_plate[row['car_id']]['license_number']
                    car_x1, car_y1, car_x2, car_y2 = [int(v) for v in car_bbox]
                    license_x1, license_y1, license_x2, license_y2 = [int(v) for v in license_plate_bbox]

                    frame = draw_border(frame, (car_x1, car_y1), (car_x2, car_y2))

                    try:
                        text_width, H = cv2.getTextSize(row['license_number'], cv2.FONT_HERSHEY_SIMPLEX, 3, 2)
                        cv2.putText(frame, row['license_number'],
                                    (int((car_x2 + car_x1 - text_width) / 2), int(car_y1) - H - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2)
                    except:
                        pass
            except (ValueError, SyntaxError):
                # Handle the case where the bbox data cannot be parsed
                continue

        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()