#  MIT License
#
#  Copyright (c) 2020 Omer Ferhat Sarioglu
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

beat_annots = {"N": "Normal beat",
               "L": "Left bundle branch block beat",
               "R": "Right bundle branch block beat",
               "B": "Bundle branch block beat (unspecified)",
               "A": "Atrial premature beat",
               "a": "Aberrated atrial premature beat",
               "J": "Nodal (junctional) premature beat}",
               "S": "Supraventricular premature or ectopic beat (atrial or nodal)",
               "V": "Premature ventricular contraction",
               "r": "R-on-T premature ventricular contraction",
               "F": "Fusion of ventricular and normal beat",
               "e": "Atrial escape beat",
               "j": "Nodal (junctional) escape beat",
               "n": "Supraventricular escape beat (atrial or nodal)",
               "E": "Ventricular escape beat",
               "/": "Paced beat",
               "f": "Fusion of paced and normal beat",
               "Q": "Unclassifiable beat",
               "?": "Beat not classified during learning"}

non_beat_annots = {"[": "Start of ventricular flutter/fibrillation",
                   "!": "Ventricular flutter wave",
                   "]": "End of ventricular flutter/fibrillation",
                   "x": "Non-conducted P-wave (blocked APC)",
                   "(": "Waveform onset",
                   ")": "Waveform end",
                   "p": "Peak of P-wave",
                   "t": "Peak of T-wave",
                   "u": "Peak of U-wave",
                   "`": "PQ junction",
                   "'": "J-point",
                   "^": "(Non-captured) pacemaker artifact",
                   "|": "Isolated QRS-like artifact",
                   "~": "Change in signal quality",
                   "+": "Rhythm change",
                   "s": "ST segment change",
                   "T": "T-wave change",
                   "*": "Systole",
                   "D": "Diastole",
                   "=": "Measurement annotation",
                   '"': "Comment annotation",
                   "@": "Link to external data"}

rythm_annots = {"AB": "Atrial bigeminy",
                "AFIB": "Atrial fibrillation",
                "AFL": "Atrial flutter",
                "B": "Ventricular bigeminy",
                "BII": "2° heart block",
                "IVR": "Idioventricular rhythm",
                "N": "Normal sinus rhythm",
                "NOD": "Nodal (A-V junctional) rhythm",
                "P": "Paced rhythm",
                "PREX": "Pre-excitation (WPW)",
                "SBR": "Sinus bradycardia",
                "SVTA": "Supraventricular tachyarrhythmia",
                "T": "Ventricular trigeminy",
                "VFL": "Ventricular flutter",
                "VT": "Ventricular tachycardia"}

TOTAL_BEAT = len(beat_annots)
TOTAL_NON_BEAT = len(non_beat_annots)
TOTAL_RYTHM = len(rythm_annots)
