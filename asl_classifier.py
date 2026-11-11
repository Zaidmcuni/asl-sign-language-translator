"""
ASL (American Sign Language) Alphabet Classifier & Gesture Engine
McGill SE Project - Comprehensive Real-Time Sign Recognition
Translates hand landmarks into ASL letters (A-Z) and interactive controls.
"""

import math
import numpy as np

def dist_3d(p1, p2):
    """Euclidean distance between two 3D points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def joint_angle(a, b, c):
    """Computes the angle in degrees at joint b formed by vectors ba and bc."""
    ba = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    norm_ba = math.sqrt(ba[0]**2 + ba[1]**2 + ba[2]**2)
    norm_bc = math.sqrt(bc[0]**2 + bc[1]**2 + bc[2]**2)
    if norm_ba * norm_bc < 1e-7:
        return 0.0
    val = max(-1.0, min(1.0, dot / (norm_ba * norm_bc)))
    return math.degrees(math.acos(val))

class ASLClassifier:
    """
    Classifies 21 3D hand landmarks into American Sign Language letters (A-Z)
    and control gestures (SPACE, DEL).
    """
    def __init__(self):
        # Human-readable guide and feedback for each sign
        self.letter_guide = {
            'A': ("Fist with upright thumb", "Thumb alongside index knuckle, fingers curled"),
            'B': ("Open 4 fingers up", "4 fingers extended straight up together, thumb folded"),
            'C': ("Curved hand 'C'", "All fingers & thumb curved into 'C' shape"),
            'D': ("Index pointing UP", "Index extended straight up, thumb touches middle tip"),
            'E': ("Curled fingers over thumb", "Fingers curled tight down, thumb tucked underneath"),
            'F': ("OK sign / 3 fingers up", "Thumb & Index tips touching, other 3 extended up"),
            'G': ("Pointing sideways", "Hand horizontal, Index & Thumb pointing forward"),
            'H': ("Two fingers horizontal", "Hand horizontal, Index & Middle extended together"),
            'I': ("Pinky straight UP", "Pinky extended up, other fingers curled into fist"),
            'K': ("Index up, Middle forward", "Index up, Middle angled up/forward, Thumb between"),
            'L': ("L-shape", "Index up & Thumb out at 90 degrees, others curled"),
            'M': ("Thumb under 3 fingers", "Thumb tucked under Index, Middle, Ring fingers"),
            'N': ("Thumb under 2 fingers", "Thumb tucked under Index & Middle fingers"),
            'O': ("O-ring shape", "All 5 fingertips touching in an O circle"),
            'P': ("Pointing down (K shape)", "Hand tilted down, Index forward, Middle down"),
            'Q': ("Pointing down (G shape)", "Hand tilted down, Index & Thumb pointing downward"),
            'R': ("Crossed fingers", "Index & Middle extended up and crossed over"),
            'S': ("Fist with thumb across", "Fingers curled tight, thumb folded ACROSS knuckles"),
            'T': ("Thumb between Index & Mid", "Thumb tucked between Index and Middle knuckles"),
            'U': ("Two fingers UP together", "Index & Middle extended up pressed side-by-side"),
            'V': ("Peace sign / Victory", "Index & Middle extended up and SPREAD apart"),
            'W': ("3 fingers UP (W shape)", "Index, Middle, Ring extended up spread apart"),
            'X': ("Index hook", "Index finger bent like a hook/claw, others curled"),
            'Y': ("Hang Loose / Shaka", "Thumb & Pinky extended out, others curled"),
            'SPACE': ("Open palm", "Show flat open palm with all fingers spread"),
            'DEL': ("Thumbs down", "Fist with thumb pointing downwards to backspace")
        }

    def extract_features(self, landmarks):
        """
        Extracts invariant geometric features from 21 MediaPipe hand landmarks.
        Works independently of distance to camera, hand size, and position.
        """
        pts = []
        for lm in landmarks:
            if hasattr(lm, 'x'):
                pts.append((lm.x, lm.y, lm.z))
            else:
                pts.append((lm[0], lm[1], lm[2]))

        # Landmark Indices:
        # 0: Wrist
        # 1: Thumb CMC, 2: Thumb MCP, 3: Thumb IP, 4: Thumb TIP
        # 5: Index MCP, 6: Index PIP, 7: Index DIP, 8: Index TIP
        # 9: Middle MCP, 10: Middle PIP, 11: Middle DIP, 12: Middle TIP
        # 13: Ring MCP, 14: Ring PIP, 15: Ring DIP, 16: Ring TIP
        # 17: Pinky MCP, 18: Pinky PIP, 19: Pinky DIP, 20: Pinky TIP

        # Palm scale S: wrist (0) to middle MCP (9)
        palm_scale = dist_3d(pts[0], pts[9])
        if palm_scale < 1e-4:
            palm_scale = 0.2

        # Palm vector from wrist (0) to middle MCP (9)
        # In screen coords: y=0 top, y=1 bottom.
        # Pointing UP means pts[9][1] < pts[0][1] (negative dy).
        dy_palm = pts[9][1] - pts[0][1]
        dx_palm = pts[9][0] - pts[0][0]

        is_pointing_up = dy_palm < -0.35 * palm_scale
        is_pointing_down = dy_palm > 0.35 * palm_scale
        is_horizontal = abs(dy_palm) < 0.45 * palm_scale and abs(dx_palm) > 0.40 * palm_scale

        # Finger definitions
        fingers = {
            'index': (5, 6, 7, 8),
            'middle': (9, 10, 11, 12),
            'ring': (13, 14, 15, 16),
            'pinky': (17, 18, 19, 20)
        }

        extended = {}
        curled = {}
        pip_angles = {}
        dip_angles = {}
        tip_mcp_dists = {}
        tip_wrist_dists = {}

        for name, (mcp, pip, dip, tip) in fingers.items():
            ang_pip = joint_angle(pts[mcp], pts[pip], pts[dip])
            ang_dip = joint_angle(pts[pip], pts[dip], pts[tip])
            d_tip_mcp = dist_3d(pts[tip], pts[mcp]) / palm_scale
            d_tip_wrist = dist_3d(pts[tip], pts[0]) / palm_scale
            d_pip_wrist = dist_3d(pts[pip], pts[0]) / palm_scale

            pip_angles[name] = ang_pip
            dip_angles[name] = ang_dip
            tip_mcp_dists[name] = d_tip_mcp
            tip_wrist_dists[name] = d_tip_wrist

            # Finger is extended if joints are straight (>130°) and tip is far from wrist/MCP
            is_ext = (ang_pip > 130 and ang_dip > 115 and d_tip_wrist > d_pip_wrist + 0.12 and d_tip_mcp > 0.82)
            # Finger is curled if tip is close to MCP/palm or joints are tightly bent (<110°)
            is_crl = (d_tip_mcp < 0.72 or ang_pip < 112 or d_tip_wrist < d_pip_wrist + 0.05)

            extended[name] = is_ext
            curled[name] = is_crl

        # Thumb features
        # 0: Wrist, 1: CMC, 2: MCP, 3: IP, 4: TIP
        thumb_ang = joint_angle(pts[0], pts[2], pts[4])
        thumb_ip_ang = joint_angle(pts[2], pts[3], pts[4])
        thumb_tip_wrist = dist_3d(pts[4], pts[0]) / palm_scale
        thumb_tip_index_mcp = dist_3d(pts[4], pts[5]) / palm_scale
        thumb_tip_pinky_mcp = dist_3d(pts[4], pts[17]) / palm_scale

        # Thumb extended away from palm (like in L, Y, open palm)
        thumb_extended = (thumb_ang > 120 and thumb_tip_index_mcp > 0.60)
        # Thumb upright alongside index (like in A)
        thumb_upright = (pts[4][1] < pts[2][1] and thumb_tip_index_mcp < 0.58 and not thumb_extended)
        # Thumb folded across the front of fingers (like in S, B)
        thumb_folded_across = (thumb_tip_pinky_mcp < 0.85 and thumb_tip_wrist < 1.15 and not thumb_extended)

        # Distances from thumb tip to fingertips
        d_thumb_index = dist_3d(pts[4], pts[8]) / palm_scale
        d_thumb_middle = dist_3d(pts[4], pts[12]) / palm_scale
        d_thumb_ring = dist_3d(pts[4], pts[16]) / palm_scale
        d_thumb_pinky = dist_3d(pts[4], pts[20]) / palm_scale

        # Inter-finger distances
        d_index_middle = dist_3d(pts[8], pts[12]) / palm_scale
        d_middle_ring = dist_3d(pts[12], pts[16]) / palm_scale
        d_ring_pinky = dist_3d(pts[16], pts[20]) / palm_scale

        # Crossed fingers for 'R'
        mcp_dx = pts[9][0] - pts[5][0]
        tip_dx = pts[12][0] - pts[8][0]
        fingers_crossed_R = (mcp_dx * tip_dx < 0) and (d_index_middle < 0.38)

        # Index hooked for 'X'
        index_hooked = (pip_angles['index'] < 130 and pip_angles['index'] > 60 and 
                        0.38 < tip_mcp_dists['index'] < 0.85 and 
                        curled['middle'] and curled['ring'] and curled['pinky'])

        # 'C' shape: fingers curved in an arc, thumb opposed
        all_curved_C = (
            85 < pip_angles['index'] < 155 and
            85 < pip_angles['middle'] < 155 and
            85 < pip_angles['ring'] < 155 and
            0.50 < d_thumb_index < 1.20 and
            d_thumb_index > 0.40
        )

        # 'O' shape: all fingertips touching thumb tip
        all_touching_O = (
            d_thumb_index < 0.40 and
            d_thumb_middle < 0.42 and
            d_thumb_ring < 0.48
        )

        return {
            'pts': pts,
            'palm_scale': palm_scale,
            'is_pointing_up': is_pointing_up,
            'is_pointing_down': is_pointing_down,
            'is_horizontal': is_horizontal,
            'extended': extended,
            'curled': curled,
            'pip_angles': pip_angles,
            'dip_angles': dip_angles,
            'tip_mcp_dists': tip_mcp_dists,
            'tip_wrist_dists': tip_wrist_dists,
            'thumb_ang': thumb_ang,
            'thumb_extended': thumb_extended,
            'thumb_upright': thumb_upright,
            'thumb_folded_across': thumb_folded_across,
            'd_thumb_index': d_thumb_index,
            'd_thumb_middle': d_thumb_middle,
            'd_thumb_ring': d_thumb_ring,
            'd_thumb_pinky': d_thumb_pinky,
            'd_index_middle': d_index_middle,
            'd_middle_ring': d_middle_ring,
            'd_ring_pinky': d_ring_pinky,
            'fingers_crossed_R': fingers_crossed_R,
            'index_hooked': index_hooked,
            'all_curved_C': all_curved_C,
            'all_touching_O': all_touching_O
        }

    def classify(self, landmarks):
        """
        Takes 21 hand landmarks and returns: (letter_str, confidence_float)
        """
        f = self.extract_features(landmarks)

        ext = f['extended']
        crl = f['curled']
        pts = f['pts']
        s = f['palm_scale']

        num_ext = sum([1 for v in ext.values() if v])
        num_crl = sum([1 for v in crl.values() if v])

        pointing_up = f['is_pointing_up']
        pointing_down = f['is_pointing_down']
        horizontal = f['is_horizontal']

        scores = {}

        # ---------------- GESTURES ----------------
        # SPACE: Flat open hand facing camera with all 5 fingers extended
        if num_ext == 4 and f['thumb_extended'] and pointing_up:
            scores['SPACE'] = 0.90

        # DEL (Backspace): Thumbs down
        if num_crl == 4 and (pts[4][1] > pts[2][1] + 0.18 * s) and not f['thumb_extended']:
            scores['DEL'] = 0.88

        # ---------------- LETTERS ----------------
        # L: Index UP, Thumb OUT (~90°), others curled
        if ext['index'] and not ext['middle'] and not ext['ring'] and not ext['pinky']:
            if f['thumb_extended'] and pointing_up:
                scores['L'] = 0.96

        # Y: Thumb & Pinky extended OUT, Index/Middle/Ring curled
        if ext['pinky'] and not ext['index'] and not ext['middle'] and not ext['ring']:
            if f['thumb_extended']:
                scores['Y'] = 0.96

        # I: Only Pinky extended UP, thumb folded
        if ext['pinky'] and crl['index'] and crl['middle'] and crl['ring'] and not f['thumb_extended']:
            if pointing_up:
                scores['I'] = 0.94

        # D: Index extended straight UP, middle/ring/pinky curled
        if ext['index'] and crl['middle'] and crl['ring'] and crl['pinky'] and not f['thumb_extended']:
            if pointing_up and not f['index_hooked']:
                scores['D'] = 0.92

        # X: Index hooked (claw shape), others curled
        if f['index_hooked']:
            scores['X'] = 0.90

        # V vs U vs R: Index & Middle UP, Ring & Pinky curled
        if ext['index'] and ext['middle'] and crl['ring'] and crl['pinky'] and not f['thumb_extended']:
            if pointing_up:
                if f['fingers_crossed_R']:
                    scores['R'] = 0.92
                elif f['d_index_middle'] > 0.40:
                    scores['V'] = 0.94
                else:
                    scores['U'] = 0.91

        # W: Index, Middle, Ring UP (3 fingers up), Pinky curled, Thumb folded
        if ext['index'] and ext['middle'] and ext['ring'] and crl['pinky'] and not f['thumb_extended']:
            if pointing_up:
                scores['W'] = 0.94

        # B: 4 fingers UP together, Thumb folded across palm
        if ext['index'] and ext['middle'] and ext['ring'] and ext['pinky']:
            if pointing_up and not f['thumb_extended']:
                scores['B'] = 0.95

        # F: Thumb & Index tips touching (OK sign), other 3 fingers extended UP
        if f['d_thumb_index'] < 0.35 and ext['middle'] and ext['ring'] and ext['pinky']:
            scores['F'] = 0.94

        # K: Index UP, Middle pointing up/forward, Thumb between them
        if ext['index'] and (ext['middle'] or f['pip_angles']['middle'] > 115) and crl['ring'] and crl['pinky']:
            d_thumb_mid_pip = dist_3d(pts[4], pts[10]) / s
            if d_thumb_mid_pip < 0.55 and pointing_up:
                scores['K'] = 0.89

        # C: All fingers curved in an open arc
        if f['all_curved_C'] and not f['all_touching_O']:
            scores['C'] = 0.88

        # O: All fingertips touching thumb tip in a circle
        if f['all_touching_O']:
            scores['O'] = 0.92

        # Sideways letters: G, H
        if horizontal:
            if ext['index'] and not ext['middle'] and crl['ring'] and crl['pinky']:
                scores['G'] = 0.89
            elif ext['index'] and ext['middle'] and crl['ring'] and crl['pinky']:
                scores['H'] = 0.89

        # Downward letters: P, Q
        if pointing_down:
            if ext['index'] and ext['middle'] and crl['ring'] and crl['pinky']:
                scores['P'] = 0.87
            elif ext['index'] and not ext['middle'] and crl['ring'] and crl['pinky']:
                scores['Q'] = 0.87

        # Fist letters: A, S, E, T, M, N (all 4 fingers curled)
        if num_crl >= 3 and not ext['index'] and not ext['middle'] and not ext['ring'] and not ext['pinky']:
            # A: Thumb straight upright alongside index knuckle
            if f['thumb_upright'] or (pts[4][1] < pts[6][1] and pts[4][0] < pts[5][0]):
                scores['A'] = 0.92
            # S: Thumb folded across knuckles in front of fingers
            elif f['thumb_folded_across'] and (pts[4][1] > pts[6][1] - 0.1 * s):
                scores['S'] = 0.89
            # E: Thumb tucked tight below fingertips
            elif f['d_thumb_index'] < 0.45 and f['d_thumb_middle'] < 0.45 and f['d_thumb_ring'] < 0.55:
                scores['E'] = 0.86
            # T: Thumb between index and middle knuckles
            elif dist_3d(pts[4], pts[6]) / s < 0.42:
                scores['T'] = 0.84
            # N: Thumb under index & middle fingers
            elif dist_3d(pts[4], pts[10]) / s < 0.42:
                scores['N'] = 0.82
            # M: Thumb under index, middle, ring fingers
            elif dist_3d(pts[4], pts[14]) / s < 0.42:
                scores['M'] = 0.81
            else:
                scores['A'] = 0.78

        if not scores:
            return None, 0.0

        best_letter = max(scores, key=scores.get)
        return best_letter, scores[best_letter]
