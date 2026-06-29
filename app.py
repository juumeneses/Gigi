# -*- coding: utf-8 -*-
from __future__ import annotations

import runpy
from pathlib import Path


runpy.run_path(str(Path(__file__).resolve().parent / "Painel_Geral" / "app.py"), run_name="__main__")
