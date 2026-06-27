# -*- coding: utf-8 -*-
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parent/'01_extrair_landmarks_faces.py'), run_name='__main__')
