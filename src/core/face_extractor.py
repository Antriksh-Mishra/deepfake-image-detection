"""
Face Extractor and Alignment Module.
Detects faces using OpenCV multi-scale cascades with robust fallback to full-frame center crop,
ensuring zero crashes and seamless handling of both raw portraits and pre-cropped face datasets.
"""

from typing import List, Optional, Tuple
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class FaceExtractor:
    def __init__(self, target_size: Tuple[int, int] = (256, 256), margin_ratio: float = 0.25):
        self.target_size = target_size
        self.margin_ratio = margin_ratio
        self.face_cascade = None
        self.eye_cascade = None
        
        # Safely attempt to load OpenCV Haar Cascades
        if HAS_CV2 and hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"
            try:
                self.face_cascade = cv2.CascadeClassifier(face_path)
            except Exception:
                self.face_cascade = None
            try:
                self.eye_cascade = cv2.CascadeClassifier(eye_path)
            except Exception:
                self.eye_cascade = None

    def _is_valid_face_candidate(self, crop_np: np.ndarray, max_face_area: float = 0) -> bool:
        """
        Validates whether a detected candidate bounding box is a genuine human face
        or a false positive from clothing text, wrinkles, or background clutter.
        """
        h, w = crop_np.shape[:2]
        if h < 50 or w < 50:
            return False
            
        # Relative area check: if a prominent face is present, discard tiny background artifacts
        if max_face_area > 80000 and (w * h) < (0.18 * max_face_area):
            return False
            
        # Aspect ratio check (human faces are roughly 1:1)
        aspect = w / (h + 1e-6)
        if not (0.70 <= aspect <= 1.40):
            return False

        # Detect desaturated grayscale images (Cr=Cb=128 causes skin_ratio=0)
        rgb_i = crop_np.astype(np.int16)
        is_grayscale = (
            float(np.mean(np.abs(rgb_i[:, :, 0] - rgb_i[:, :, 1]))) < 3.0
            and float(np.mean(np.abs(rgb_i[:, :, 0] - rgb_i[:, :, 2]))) < 3.0
        )

        # Skin-tone presence in YCrCb color space
        try:
            ycrcb = cv2.cvtColor(crop_np, cv2.COLOR_RGB2YCrCb)
            cr = ycrcb[:, :, 1]
            cb = ycrcb[:, :, 2]
            skin_mask = (cr >= 130) & (cr <= 180) & (cb >= 70) & (cb <= 135)
            skin_ratio = float(np.mean(skin_mask))
        except Exception:
            skin_ratio = 0.50

        # Unconditional skin minimum on color crops blocks clothing text/logos even if eye Haar misfires
        if not is_grayscale and skin_ratio < 0.14:
            return False

        # Eye detection in upper 65% of crop (minNeighbors=3 curbs high-contrast text false hits)
        has_eyes = False
        if self.eye_cascade is not None and not self.eye_cascade.empty():
            try:
                gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
                upper_gray = gray[:int(h * 0.65), :]
                eyes = self.eye_cascade.detectMultiScale(upper_gray, scaleFactor=1.1, minNeighbors=3)
                has_eyes = len(eyes) > 0
            except Exception:
                has_eyes = False

        if is_grayscale:
            return has_eyes

        # Color candidate needs eye confirmation with >=14% skin OR strong skin presence (>=24%)
        if not ((has_eyes and skin_ratio >= 0.14) or skin_ratio >= 0.24):
            return False

        return True

    def detect_faces(self, image_np: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects bounding boxes (x, y, w, h) for all genuine faces in image_np (RGB).
        Filters out false positives on clothing text, folds, and background clutter.
        """
        if self.face_cascade is None:
            return []
            
        try:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            gray = cv2.equalizeHist(gray)
            
            min_dim = min(image_np.shape[:2])
            min_face = max(45, int(min_dim * 0.07))
            
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(min_face, min_face),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) == 0:
                return []
                
            max_area = max(fw * fh for fx, fy, fw, fh in faces)
            valid_faces = []
            for fx, fy, fw, fh in faces:
                crop = image_np[fy:fy + fh, fx:fx + fw]
                if self._is_valid_face_candidate(crop, max_face_area=max_area):
                    valid_faces.append((int(fx), int(fy), int(fw), int(fh)))
                    
            return valid_faces
        except Exception:
            return []

    def detect_object_bounds(self, image_np: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Detects the salient foreground object bounding box (x, y, w, h) for non-face images
        such as shoes, vehicles, animals, products, and AI-generated artwork.
        Falls back cleanly to a centered square crop if no distinct foreground contour exists.
        """
        h_img, w_img = image_np.shape[:2]
        min_dim = min(h_img, w_img)
        center_fallback = ((w_img - min_dim) // 2, (h_img - min_dim) // 2, min_dim, min_dim)

        if not HAS_CV2:
            return center_fallback

        try:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            # Sample image corners to estimate background intensity (light studio vs dark background)
            corner_size = max(5, int(min(h_img, w_img) * 0.06))
            corners = [
                gray[:corner_size, :corner_size],
                gray[:corner_size, -corner_size:],
                gray[-corner_size:, :corner_size],
                gray[-corner_size:, -corner_size:],
            ]
            corner_mean = float(np.mean([np.mean(c) for c in corners]))
            is_light_bg = corner_mean > 140.0

            # Threshold to isolate foreground object from background
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            if is_light_bg:
                thresh_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                _, thresh = cv2.threshold(blurred, 0, 255, thresh_type)
            else:
                thresh_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU
                _, thresh = cv2.threshold(blurred, 0, 255, thresh_type)

            # Morphological cleanup to connect object fragments
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return center_fallback

            # Filter out tiny noise contours; find dominant object contour
            total_area = float(h_img * w_img)
            valid_contours = [c for c in contours if cv2.contourArea(c) > 0.03 * total_area]
            if not valid_contours:
                return center_fallback

            best_c = max(valid_contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(best_c)

            # Add a 10% margin around the detected object for contextual forensic evaluation
            margin_x = int(bw * 0.10)
            margin_y = int(bh * 0.10)
            ox1 = max(0, bx - margin_x)
            oy1 = max(0, by - margin_y)
            ox2 = min(w_img, bx + bw + margin_x)
            oy2 = min(h_img, by + bh + margin_y)
            return (ox1, oy1, ox2 - ox1, oy2 - oy1)
        except Exception:
            return center_fallback

    def classify_image_context(self, image_np: np.ndarray, has_face: bool) -> str:
        """
        Classifies the high-level semantic domain of the input image
        to ensure forensic terminology and explainability remain strictly domain-appropriate.
        """
        if has_face:
            return "Human Face Portrait"

        h, w = image_np.shape[:2]
        try:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if HAS_CV2 else np.mean(image_np, axis=-1)
            # Corner background uniformity check
            corner_s = max(5, int(min(h, w) * 0.08))
            corners = [
                gray[:corner_s, :corner_s],
                gray[:corner_s, -corner_s:],
                gray[-corner_s:, :corner_s],
                gray[-corner_s:, -corner_s:],
            ]
            corner_std = float(np.mean([np.std(c) for c in corners]))
            corner_mean = float(np.mean([np.mean(c) for c in corners]))

            # Product / Commercial photograph signature (clean uniform light/dark background)
            if corner_std < 18.0 and (corner_mean > 190.0 or corner_mean < 40.0):
                return "Product / Commercial Object Photograph"

            # Color saturation & edge entropy check
            if HAS_CV2:
                hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
                sat_mean = float(np.mean(hsv[:, :, 1]))
                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if sat_mean > 120.0 and lap_var > 400.0:
                    return "AI-Generated Artwork / Digital Graphic"

            return "General Object / Natural Scene"
        except Exception:
            return "General Media / Object"

    def extract_face(
        self, 
        image: Image.Image
    ) -> Tuple[Image.Image, Tuple[int, int, int, int], bool]:
        """
        Extracts the primary (largest) face with a contextual margin.
        If no face is detected, returns the object-aware center-cropped resized image.
        
        Returns:
            aligned_face: PIL Image resized to target_size
            bbox: (x, y, w, h) bounding box coordinates
            face_detected: bool indicating whether a face was found
        """
        image_np = np.array(image.convert("RGB"))
        h_img, w_img, _ = image_np.shape
        
        faces = self.detect_faces(image_np)

        # Preserve pre-cropped square portraits only when the face already dominates the square frame
        is_already_square = 0.85 <= (w_img / (h_img + 1e-6)) <= 1.18
        if len(faces) > 0:
            largest_face = max(faces, key=lambda b: b[2] * b[3])
            fx, fy, fw, fh = largest_face
            tight_cropped_square = is_already_square and (fw >= 0.58 * w_img) and (fh >= 0.58 * h_img)
            if tight_cropped_square:
                min_dim = min(h_img, w_img)
                start_x = (w_img - min_dim) // 2
                start_y = (h_img - min_dim) // 2
                crop = image_np[start_y:start_y + min_dim, start_x:start_x + min_dim]
                face_img = Image.fromarray(crop).resize(self.target_size, Image.Resampling.LANCZOS)
                return face_img, (start_x, start_y, min_dim, min_dim), True

            # Contextual square crop centered on the largest face
            fcx = fx + fw // 2
            fcy = fy + fh // 2
            box_size = int(max(fw, fh) * (1.0 + 2.0 * self.margin_ratio))

            x1 = max(0, fcx - box_size // 2)
            y1 = max(0, fcy - box_size // 2)
            x2 = min(w_img, x1 + box_size)
            y2 = min(h_img, y1 + box_size)

            real_w = x2 - x1
            real_h = y2 - y1
            sq_size = min(real_w, real_h)

            face_crop = image_np[y1:y1 + sq_size, x1:x1 + sq_size]
            face_img = Image.fromarray(face_crop).resize(self.target_size, Image.Resampling.LANCZOS)
            return face_img, (x1, y1, sq_size, sq_size), True

        # Non-face object localization fallback
        ox, oy, ow, oh = self.detect_object_bounds(image_np)
        ocx = ox + ow // 2
        ocy = oy + oh // 2
        sq_size = min(h_img, w_img, max(ow, oh, int(min(h_img, w_img) * 0.70)))
        x1 = max(0, min(w_img - sq_size, ocx - sq_size // 2))
        y1 = max(0, min(h_img - sq_size, ocy - sq_size // 2))
        crop = image_np[y1:y1 + sq_size, x1:x1 + sq_size]
        obj_img = Image.fromarray(crop).resize(self.target_size, Image.Resampling.LANCZOS)
        return obj_img, (x1, y1, sq_size, sq_size), False

    def extract_all_faces(
        self, 
        image: Image.Image
    ):
        """
        Extracts all detected faces with contextual margins.
        If no face is detected, returns an object-aware localized entry without assuming facial content.
        """
        image_np = np.array(image.convert("RGB"))
        h_img, w_img, _ = image_np.shape
        faces = self.detect_faces(image_np)
        
        if len(faces) == 0:
            ox, oy, ow, oh = self.detect_object_bounds(image_np)
            ocx = ox + ow // 2
            ocy = oy + oh // 2
            sq_size = min(h_img, w_img, max(ow, oh, int(min(h_img, w_img) * 0.70)))
            x1 = max(0, min(w_img - sq_size, ocx - sq_size // 2))
            y1 = max(0, min(h_img - sq_size, ocy - sq_size // 2))
            crop = image_np[y1:y1 + sq_size, x1:x1 + sq_size]
            obj_img = Image.fromarray(crop).resize(self.target_size, Image.Resampling.LANCZOS)
            img_context = self.classify_image_context(image_np, has_face=False)
            return [{
                "face_index": 0,
                "aligned_face": obj_img,
                "bbox": (x1, y1, sq_size, sq_size),
                "object_bbox": (ox, oy, ow, oh),
                "face_detected": False,
                "has_face": False,
                "is_primary": True,
                "area": sq_size * sq_size,
                "image_type": img_context,
            }]
            
        faces_sorted = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        results = []
        for idx, (fx, fy, fw, fh) in enumerate(faces_sorted):
            fcx = fx + fw // 2
            fcy = fy + fh // 2
            box_size = int(max(fw, fh) * (1.0 + 2.0 * self.margin_ratio))
            
            x1 = max(0, fcx - box_size // 2)
            y1 = max(0, fcy - box_size // 2)
            x2 = min(w_img, x1 + box_size)
            y2 = min(h_img, y1 + box_size)
            
            real_w = x2 - x1
            real_h = y2 - y1
            sq_size = min(real_w, real_h)
            
            face_crop = image_np[y1:y1 + sq_size, x1:x1 + sq_size]
            face_img = Image.fromarray(face_crop).resize(self.target_size, Image.Resampling.LANCZOS)
            
            results.append({
                "face_index": idx,
                "aligned_face": face_img,
                "bbox": (x1, y1, sq_size, sq_size),
                "raw_face_bbox": (int(fx), int(fy), int(fw), int(fh)),
                "object_bbox": (int(fx), int(fy), int(fw), int(fh)),
                "face_detected": True,
                "has_face": True,
                "is_primary": (idx == 0),
                "area": fw * fh,
                "image_type": "Human Face Portrait",
            })
            
        return results
