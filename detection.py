import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import model.efficientnet.models.efficientnet as eff
from utils.hungarian import iou, hungarian_stars

checkpoint = torch.load('checkpoints/best.pth', map_location=torch.device('cpu'))
model = eff.efficientnet_b0(pretrained=False, num_classes=3)
model.load_state_dict(checkpoint['model'])
model.eval()

class_names = ['background', 'chicken', 'sheep']

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

def preprocess_frame(frame, size=224, fill=128):
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    w, h = frame_pil.size
    scale = size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    frame_pil = frame_pil.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas.paste(frame_pil, (left, top))

    transform = get_transform()
    return transform(canvas).unsqueeze(0)

def classify_frame(frame):
    input_tensor = preprocess_frame(frame)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1) 
        confidence, predicted_class = torch.max(probabilities, dim=1)
    return predicted_class.item(), confidence.item()

def merge_nested_tracks(tracks, track_age, tracks_info, last_cls_frame, cover_thr=0.85):
    def area(box):
        x0, y0, x1, y1 = box
        return max(0, x1 - x0) * max(0, y1 - y0)

    def inter_area(b1, b2):
        x0 = max(b1[0], b2[0])
        y0 = max(b1[1], b2[1])
        x1 = min(b1[2], b2[2])
        y1 = min(b1[3], b2[3])
        w = max(0, x1 - x0)
        h = max(0, y1 - y0)
        return w * h

    changed = True
    while changed:
        changed = False
        ids = list(tracks.keys())

        for i in range(len(ids)):
            id1 = ids[i]
            if id1 not in tracks:
                continue

            b1 = tracks[id1]
            a1 = area(b1)
            if a1 == 0:
                continue

            c1, p1 = tracks_info.get(id1, (0, 0.0))
            if c1 == 0:
                continue

            for j in range(i + 1, len(ids)):
                id2 = ids[j]
                if id2 not in tracks:
                    continue

                b2 = tracks[id2]
                a2 = area(b2)
                if a2 == 0:
                    continue

                c2, p2 = tracks_info.get(id2, (0, 0.0))
                if c2 == 0:
                    continue

                if c1 != c2:
                    continue

                inter = inter_area(b1, b2)
                if inter == 0:
                    continue

                if a1 <= a2:
                    small_id, big_id = id1, id2
                    small_box, big_box = b1, b2
                    small_area = a1
                else:
                    small_id, big_id = id2, id1
                    small_box, big_box = b2, b1
                    small_area = a2

                cover = inter / (small_area + 1e-6)
                if cover < cover_thr:
                    continue

                new_box = [
                    min(big_box[0], small_box[0]),
                    min(big_box[1], small_box[1]),
                    max(big_box[2], small_box[2]),
                    max(big_box[3], small_box[3]),
                ]
                tracks[big_id] = new_box

                cb, pb = tracks_info.get(big_id, (0, 0.0))
                cs, ps = tracks_info.get(small_id, (0, 0.0))
                tracks_info[big_id] = (cb, max(pb, ps))

                tracks.pop(small_id, None)
                track_age.pop(small_id, None)
                tracks_info.pop(small_id, None)
                last_cls_frame.pop(small_id, None)

                changed = True
                break

            if changed:
                break
            
def main():
    video = cv2.VideoCapture("data/animals.MP4")

    ok, frame1 = video.read()
    if not ok:
        video.release()
        cv2.destroyAllWindows()
        raise SystemExit(1)

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('results/result_animals.mp4', fourcc, fps, (frame_width, frame_height))
    # bg_video = cv2.VideoWriter('result_background.mp4', fourcc, fps, (frame_width, frame_height))

    background = cv2.imread("data/background.jpg")
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY).astype(np.float32)

    alpha = 0.95
    update_every = 5
    frame_idx = 0

    tracks = {}
    track_age = {}
    tracks_info = {}
    last_cls_frame = {}
    next_id = 0

    max_age = 10
    iou_match_thr = 0.3
    cls_every_n = max(1, int(round(fps * 0.5)))

    stable = np.zeros_like(background, dtype=np.uint8)
    N = 8

    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray_frame_f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = cv2.absdiff(gray_frame_f, background).astype(np.uint8)

        diff = cv2.blur(diff, (4, 4))
        _, diff = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel, iterations=1)
        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel, iterations=2)

        bg_like = (diff == 0).astype(np.uint8)
        stable = np.where(bg_like, np.minimum(stable + 1, 255), 0)

        if frame_idx % update_every == 0:
            bg_mask = (stable >= N)
            background[bg_mask] = alpha * background[bg_mask] + (1 - alpha) * gray_frame_f[bg_mask]

        contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for el in contours:
            x, y, w, h = cv2.boundingRect(el)
            aspect = w / (h + 1e-6)
            if aspect > 3.0 or aspect < 0.33:
                continue
            if (w * h < 3000) or (w < 40) or (h < 40):
                continue
            pad = 20
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(frame_width, x + w + pad)
            y1 = min(frame_height, y + h + pad)
            detections.append([x0, y0, x1, y1])

        track_ids = list(tracks.keys())
        matched_det = set()
        matched_track = set()

        if len(track_ids) > 0 and len(detections) > 0:
            n_prev = len(track_ids)
            n_curr = len(detections)
            n = max(n_prev, n_curr)
            cost = np.ones((n, n), dtype=np.float32)
            for i in range(n_prev):
                for j in range(n_curr):
                    cost[i, j] = 1.0 - iou(tracks[track_ids[i]], detections[j])
            matches = hungarian_stars(cost)
            matches = [(r, c) for r, c in matches if r < n_prev and c < n_curr]
            for r, c in matches:
                iou_val = 1.0 - cost[r, c]
                if iou_val >= iou_match_thr:
                    tid = track_ids[r]
                    tracks[tid] = detections[c]
                    track_age[tid] = 0
                    matched_track.add(tid)
                    matched_det.add(c)

        for j, det in enumerate(detections):
            if j not in matched_det:
                tid = next_id
                next_id += 1
                tracks[tid] = det
                track_age[tid] = 0
                matched_track.add(tid)
                tracks_info[tid] = (0, 0.0)
                last_cls_frame[tid] = -10**9

        to_delete = []
        for tid in list(tracks.keys()):
            if tid not in matched_track:
                track_age[tid] += 1
                if track_age[tid] > max_age:
                    to_delete.append(tid)

        for tid in to_delete:
            tracks.pop(tid, None)
            track_age.pop(tid, None)
            tracks_info.pop(tid, None)
            last_cls_frame.pop(tid, None)

        for tid, (x0, y0, x1, y1) in tracks.items():
            if frame_idx - last_cls_frame[tid] >= cls_every_n:
                roi = frame[y0:y1, x0:x1]
                if roi.size != 0:
                    class_id, prob = classify_frame(roi)
                    tracks_info[tid] = (class_id, prob)
                    last_cls_frame[tid] = frame_idx

        merge_nested_tracks(tracks, track_age, tracks_info, last_cls_frame, cover_thr=0.7)

        for tid, (x0, y0, x1, y1) in tracks.items():
            class_id, probability = tracks_info.get(tid, (0, 0.0))
            if class_id == 0 or (class_id == 1 and probability < 0.86) or (class_id == 2 and probability < 0.9):
                continue

            roi = frame[y0:y1, x0:x1]

            label = f'ID {tid} {class_names[class_id]} {probability:.2f}'
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.putText(frame, label, (x0, max(0, y0 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        frame_idx += 1

        cv2.imshow("Detection", frame)
        cv2.imshow("Diff", diff)
        # bg_video.write(bg_vis_bgr)
        output_video.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    output_video.release()
    # bg_video.release()
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()