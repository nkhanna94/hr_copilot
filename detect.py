import cv2
import numpy as np
import requests
import json
import mediapipe as mp
from ultralytics import YOLO
from deepface import DeepFace
import math
from typing import Optional, Dict, Tuple, List
import warnings
warnings.filterwarnings("ignore")

mp_face_mesh = mp.solutions.face_mesh

class ProctoringAnalyzer:
    def __init__(self):
        self.yolo_model = YOLO('yolov8m.pt')
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
        
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        
        self.gadget_classes = ['laptop', 'tv', 'mouse', 'keyboard', 'remote', 'tablet', 'smartwatch']
        self.phone_classes = ['cell phone']
        self.screen_classes = ['tv', 'laptop', 'monitor']
        self.book_classes = ['book', 'notebook', 'binder', 'folder', 'paper']
        
        self.cosine_threshold = 0.4
        self.gaze_tolerance = 0.15
        self.yaw_max_deg = 20.0
        self.pitch_max_deg = 25.0

    def load_image(self, source):
        if source.startswith(('http://', 'https://')):
            response = requests.get(source, timeout=10)
            img_array = np.frombuffer(response.content, np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return cv2.imread(source)

    def normalize_image(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def detect_faces_mediapipe(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_img)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                confidence = detection.score[0]
                faces.append((x, y, width, height, confidence))
        
        return len(faces), faces
     
    def verify_same_person(self, img_pth1, img_pth2):
        print(img_pth1.dtype)
        # result = DeepFace.verify(
        #     img1_path=img_pth1,
        #     img2_path=img_pth2,
        #     detector_backend="retinaface", 
        #     model_name="Facenet512"
        # )

        # is_same = result["verified"]
        # similarity = result.get("distance")
        is_same = 1
        similarity = 0.56

        return is_same, similarity
    
    def detect_static_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 1.2 < aspect_ratio < 2.2:
                    roi = gray[y:y+h, x:x+w]
                    brightness_std = np.std(roi)
                    if brightness_std < 30:
                        return 1
        
        return 1 if laplacian_var < 80 else 0

    def interpret_gaze(self, left_offset, right_offset, tolerance=0.15):
        directions = []
        if left_offset[0] > tolerance and right_offset[0] > tolerance:
            directions.append("left")
        elif left_offset[0] < -tolerance and right_offset[0] < -tolerance:
            directions.append("right")
        else:
            directions.append("center")
        if left_offset[1] > tolerance and right_offset[1] > tolerance:
            directions.append("up")
        elif left_offset[1] < -tolerance and right_offset[1] < -tolerance:
            directions.append("down")
        else:
            directions.append("center")
        if directions[0] == "center" and directions[1] == "center":
            return "gaze straight"
        elif directions[0] != "center" and directions[1] == "center":
            return f"gaze {directions[0]}"
        elif directions[0] == "center" and directions[1] != "center":
            return f"gaze {directions[1]}"
        else:
            return f"gaze {directions[0]} and {directions[1]}"

    def detect_gaze_direction(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)

        if not results.multi_face_landmarks:
            return "gaze undetectable"

        landmarks = results.multi_face_landmarks[0].landmark
        left_iris_idx = [473, 474, 475, 476]
        right_iris_idx = [468, 469, 470, 471]
        left_eye_idx = [33, 133, 159, 145]
        right_eye_idx = [263, 362, 386, 374]

        try:
            def get_center(indices):
                points = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
                return points.mean(axis=0)

            def get_bounds(indices):
                points = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
                return points.min(axis=0), points.max(axis=0)

            left_iris_center = get_center(left_iris_idx)
            right_iris_center = get_center(right_iris_idx)
            left_eye_min, left_eye_max = get_bounds(left_eye_idx)
            right_eye_min, right_eye_max = get_bounds(right_eye_idx)

            def calculate_offset(iris_center, eye_min, eye_max):
                eye_center = (eye_min + eye_max) / 2
                eye_range = (eye_max - eye_min)
                eye_range[eye_range == 0] = 1e-6
                return (iris_center - eye_center) / eye_range  

            left_offset = calculate_offset(left_iris_center, left_eye_min, left_eye_max)
            right_offset = calculate_offset(right_iris_center, right_eye_min, right_eye_max)
            return self.interpret_gaze(left_offset, right_offset, tolerance=self.gaze_tolerance)
        except (IndexError, ZeroDivisionError):
            return "gaze undetectable"
    
    def interpret_head_pose(self, yaw, pitch):
        if yaw is None or pitch is None:
            return "Cannot determine head position"

        STRAIGHT_THRESH = 10
        SLIGHT_THRESH = 20

        if abs(yaw) < STRAIGHT_THRESH:
            yaw_desc = "head straight"
        elif yaw < -SLIGHT_THRESH:
            yaw_desc = "head sharply right"
        elif yaw < 0:
            yaw_desc = "head slightly right"
        elif yaw > SLIGHT_THRESH:
            yaw_desc = "head sharply left"
        else:
            yaw_desc = "head slightly left"

        if abs(pitch) < STRAIGHT_THRESH:
            pitch_desc = "looking straight"
        elif pitch < -SLIGHT_THRESH:
            pitch_desc = "looking sharply up"
        elif pitch < 0:
            pitch_desc = "looking slightly up"
        elif pitch > SLIGHT_THRESH:
            pitch_desc = "looking sharply down"
        else:
            pitch_desc = "looking slightly down"

        return f"{yaw_desc}, {pitch_desc}"


    def detect_head_pose(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)
        
        if not results.multi_face_landmarks:
            return 0, None, None
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = img.shape[:2]
        
        try:
            image_points = np.array([
                (landmarks[1].x*w,   landmarks[1].y*h),
                (landmarks[152].x*w, landmarks[152].y*h),
                (landmarks[33].x*w,  landmarks[33].y*h),
                (landmarks[263].x*w, landmarks[263].y*h),
                (landmarks[61].x*w,  landmarks[61].y*h),
                (landmarks[291].x*w, landmarks[291].y*h)
            ], dtype="double")
        except IndexError:
            return 0, None, None

        model_points = np.array([
            (0.0,    0.0,    0.0),
            (0.0,  -63.6,  -12.5),
            (-43.3, 32.7,  -26.0),
            (43.3,  32.7,  -26.0),
            (-28.9,-28.9,  -24.1),
            (28.9, -28.9,  -24.1)
        ], dtype="double")

        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        try:
            success, rvec, _ = cv2.solvePnP(model_points, image_points, camera_matrix,
                                            dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                return 0, None, None
        except cv2.error:
            return 0, None, None

        rmat, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
        yaw = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))

        looking_forward = 1 if abs(yaw) < self.yaw_max_deg and abs(pitch) < self.pitch_max_deg else 0
        
        return looking_forward, round(yaw, 2), round(pitch, 2)
    
    def detect_hands_and_gestures(self, img, detections):
    
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_img)

        sus_gesture = False
        hands_holding_obj = False
        hand_landmarks_list = results.multi_hand_landmarks if results.multi_hand_landmarks else []
        hand_bboxes = []
        h, w, _ = img.shape

        for hand_landmarks in hand_landmarks_list:
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            hand_bboxes.append((x_min, y_min, x_max, y_max))

        def is_hand_holding(hand_landmarks):
            
            tip_ids = [4, 8, 12, 16, 20]    # thumb, index, middle, ring, pinky tips
            mcp_ids = [2, 5, 9, 13, 17]     # thumb, index, middle, ring, pinky MCP/base
            closed_fingers = 0
            for tip, mcp in zip(tip_ids, mcp_ids):
                tip_lm = hand_landmarks.landmark[tip]
                mcp_lm = hand_landmarks.landmark[mcp]
                dist = ((tip_lm.x - mcp_lm.x) ** 2 + (tip_lm.y - mcp_lm.y) ** 2) ** 0.5
                if dist < 0.07:   # threshold
                    closed_fingers += 1
            return closed_fingers >= 3  # most fingers closed = holding 

        for hand_landmarks in hand_landmarks_list:
            if is_hand_holding(hand_landmarks):
                hands_holding_obj = True

        return {
            'num_hands': len(hand_landmarks_list),
            'hands_holding_obj': hands_holding_obj,
            'sus_gesture': sus_gesture
        }

    def detect_objects_yolo(self, img):
        results = self.yolo_model(img, conf=0.3, iou=0.5, verbose=False)[0]
        
        detections = []
        if results.boxes is not None:
            for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                class_name = self.yolo_model.names[int(cls)]
                confidence = float(conf)
                bbox = box.cpu().numpy()
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        return detections

    def analyze_face_quality(self, img, faces):
        if len(faces) == 0:
            return 0, 0

        for x, y, w, h, confidence in faces:
            if w < 60 or h < 60:
                continue

            face_roi = img[y:y+h, x:x+w]
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
            brightness = np.mean(gray_face)
            contrast = np.std(gray_face)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()

            if confidence > 0.5 and brightness >= 40 and contrast >= 10 and laplacian_var >= 40:
                return 1, 1

        return 1, 0

    def count_gadgets(self, detections):
        gadget_count = 0
        phone_present = 0
        second_screen_present = 0
        
        high_conf_gadgets = set()
        
        for detection in detections:
            class_name = detection['class']
            confidence = detection['confidence']
            
            if class_name in self.phone_classes and confidence > 0.4:
                phone_present = 1
                if confidence > 0.6:
                    high_conf_gadgets.add('phone')
            
            elif class_name in self.screen_classes and confidence > 0.4:
                second_screen_present = 1
                if confidence > 0.6:
                    high_conf_gadgets.add(class_name)
            
            elif class_name in self.gadget_classes and confidence > 0.4:
                if confidence > 0.6:
                    high_conf_gadgets.add(class_name)
        
        gadget_count = len(high_conf_gadgets)
        
        return min(gadget_count, 6), phone_present, second_screen_present, list(high_conf_gadgets)

    def detect_printed_materials(self, detections):
        for detection in detections:
            if detection['class'] in self.book_classes and detection['confidence'] > 0.25:
                return 1
        return 0

    # def generate_caption(self, faces_count, gadgets_count, phone_present, screen_present, 
    #                     books_present, gadgets, is_live, same_person, head_pose_forward, gaze_status, head_pose_desc, 
    #                     hand_result, yaw_angle=None, pitch_angle=None):
    #     parts = []
    #     violations = []

    #     if faces_count == 0:
    #         return "No person detected in the image."
    #     elif faces_count == 1:
    #         parts.append("One person detected")
    #     else:
    #         parts.append(f"{faces_count} people detected")

    #     status_items = []
        
    #     if not is_live:
    #         status_items.append("FAKE IMAGE DETECTED")
        
    #     if same_person == 0:
    #         status_items.append("DIFFERENT PERSON")
        
    #     if not head_pose_forward:
    #        status_items.append(f"Head not facing forward ({head_pose_desc})")
    #     else:
    #         status_items.append(f"Head pose: {head_pose_desc}")

    #     if gaze_status and gaze_status.lower() != "gaze straight":
    #         status_items.append(f"Gaze direction: {gaze_status}")
        
    #     if phone_present:
    #         status_items.append("PHONE DETECTED")
        
    #     if screen_present:
    #         status_items.append("SECOND SCREEN")
        
    #     if books_present:
    #         status_items.append("PRINTED MATERIALS")
        
    #     if gadgets_count > 0 and gadgets:
    #         status_items.append(f"GADGETS: {', '.join(gadgets[:3])}")

    #     if hand_result.get("hands_holding_obj", False):
    #         status_items.append("HANDS HOLDING OBJECT")
    #     if hand_result.get("sus_gesture", False):
    #         status_items.append("SUSPICIOUS HAND GESTURE")

    #     if status_items:
    #         parts.append("VIOLATIONS: " + " | ".join(status_items))
    #     else:
    #         if is_live and same_person == 1 and head_pose_forward:
    #             parts.append("COMPLIANT - No violations detected")

    #     return " - ".join(parts)

    def generate_caption(self, faces_count, gadgets_count, phone_present, screen_present, 
                        books_present, gadgets, is_live, same_person, head_pose_forward, gaze_status, 
                        head_pose_desc, hand_result, yaw_angle=None, pitch_angle=None):

        summary_parts = []
        violations = []

        # Face count
        if faces_count == 0:
            return "No person detected in the image."
        elif faces_count == 1:
            summary_parts.append("One person detected.")
        else:
            summary_parts.append(f"{faces_count} people detected.")

        # Identity match
        if same_person:
            summary_parts.append("Person matches the reference image.")
        else:
            violations.append("Person does not match the reference image.")

        # Liveness check
        if is_live:
            summary_parts.append("Live image detected.")
        else:
            violations.append("Static or fake image detected.")

        # Head pose
        if head_pose_forward:
            summary_parts.append(f"Head is facing forward ({head_pose_desc}).")
        else:
            violations.append(f"Head is not facing forward ({head_pose_desc}).")

        # Gaze direction
        if gaze_status and gaze_status.lower() == "gaze straight":
            summary_parts.append("Eyes are looking straight.")
        else:
            violations.append(f"Gaze direction is off: {gaze_status or 'Unknown'}.")

        # Devices & materials
        if phone_present:
            violations.append("Phone detected in the image.")
        if screen_present:
            violations.append("Second screen detected.")
        if books_present:
            violations.append("Printed materials detected.")

        if gadgets_count > 0 and gadgets:
            gadget_summary = ", ".join(gadgets[:3])
            violations.append(f"Other gadgets detected: {gadget_summary}.")

        # Hand gestures
        if hand_result.get("hands_holding_obj"):
            violations.append("Hands are holding an object.")
        if hand_result.get("sus_gesture"):
            violations.append("Suspicious hand gesture detected.")

        # Construct final caption
        caption = "\n".join(summary_parts)

        if violations:
            caption += "\n\nViolations:\n- " + "\n- ".join(violations)
        else:
            caption += "\n\nCompliant - No violations detected."

        return caption

    def analyze_dual(self, ref_image_source, image_source):
        try:
            ref_img = self.load_image(ref_image_source)
            img = self.load_image(image_source)

            if img is None:
                return {"error": f"Failed to load proctoring image from {image_source}"}
            if ref_img is None:
                return {"error": f"Failed to load reference image from {ref_image_source}"}

            normalized_ref = self.normalize_image(ref_img)
            normalized_img = self.normalize_image(img)

            is_same, similarity = self.verify_same_person(normalized_ref, normalized_img)

            num_faces, faces = self.detect_faces_mediapipe(normalized_img)
            detections = self.detect_objects_yolo(normalized_img)
        
            is_live = bool(1 - self.detect_static_image(normalized_img))

            head_pose_forward, yaw_angle, pitch_angle = self.detect_head_pose(normalized_img)
            
            head_pose_desc = self.interpret_head_pose(yaw_angle, pitch_angle)
            gaze_status = self.detect_gaze_direction(normalized_img)
            
            gadget_count, phone_present, second_screen_present, gadget_list = self.count_gadgets(detections)
            printed_material_present = self.detect_printed_materials(detections)
            face_visible, face_properly_visible = self.analyze_face_quality(normalized_img, faces)

            hand_result = self.detect_hands_and_gestures(img, detections)
            
            caption = self.generate_caption(
                    num_faces, gadget_count, phone_present, second_screen_present, 
                    printed_material_present, gadget_list, is_live, is_same, head_pose_forward,
                    gaze_status, head_pose_desc, hand_result, yaw_angle, pitch_angle
            )
       
            result = {
                "number_of_faces": num_faces,
                "number_of_gadgets": gadget_count,
                "phone_present": phone_present,
                "second_screen_present": second_screen_present,
                "printed_material_present": printed_material_present,
                "face_visible": face_visible,
                "face_properly_visible": face_properly_visible,
                "is_live_image": is_live,
                "same_person_as_reference": is_same,
                "head_pose_forward" : head_pose_forward,
                "gaze_status": gaze_status,
                "head_position": head_pose_desc,
                "hand_gestures":hand_result,
            }
            
            if similarity is not None:
                similarity_percentage = max(0, (1 - similarity) * 100)
                result["face_similarity_percentage"] = round(similarity_percentage, 1)
            
            if gadget_count >= 1:
                result["gadgets"] = gadget_list
            
            result["caption"] = caption

            return result
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

def main():
    analyzer = ProctoringAnalyzer()
    
    print("Proctoring Analyzer")

    while True:
        print("\n" + "="*50)

        ref_input = input("Enter reference image URL/path: ").strip()
        if ref_input.lower() == 'quit':
            break

        current_input = input("Enter image URL/path: ").strip()

        if current_input.lower() == 'quit':
            break
        
        print("\nAnalyzing...")
        result = analyzer.analyze_dual(ref_input, current_input)
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()