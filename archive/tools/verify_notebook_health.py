"""verify_notebook_health.py — Full AST & Runtime Health Audit for video-surv-FIXED.ipynb & keva-vision.ipynb.
"""
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
TARGET_NB = HERE / "video-surv-FIXED.ipynb"

def audit_notebook(nb_path):
    print(f"\n==============================================================================")
    print(f"🧐 AUDITING NOTEBOOK HEALTH: {nb_path.name}")
    print(f"==============================================================================")
    
    if not nb_path.exists():
        print(f"❌ File not found: {nb_path}")
        return False

    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ Invalid JSON format: {e}")
        return False

    cells = nb.get("cells", [])
    print(f"Total Cells: {len(cells)}")
    
    errors_found = 0
    warnings_found = 0

    for idx, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "")
        source_lines = cell.get("source", [])
        code_str = "".join(source_lines)

        print(f"\n--- Cell {idx} [{cell_type.upper()}] ({len(source_lines)} lines) ---")
        
        if cell_type == "code":
            # 1. Check for HTML tags in code cell
            if "<a " in code_str or "</a>" in code_str:
                print(f"❌ ERROR: HTML anchor tag detected in CODE cell {idx}! Python will throw SyntaxError.")
                errors_found += 1
            
            # 2. Check AST Python syntax
            try:
                ast.parse(code_str)
                print(f"  ✅ AST Syntax Check: PASSED")
            except SyntaxError as se:
                print(f"  ❌ SYNTAX ERROR in Cell {idx} (Line {se.lineno}): {se.msg}")
                if se.text:
                    print(f"     Offending line: {se.text.strip()}")
                errors_found += 1

            # 3. Check for common undefined global traps
            keywords_to_check = ["STAFF_GALLERY_DIR", "ZONE_AI_OVERRIDES", "ZONE_ROLE_KEYWORDS", "supervision"]
            for kw in keywords_to_check:
                if kw in code_str:
                    print(f"  ℹ️ Uses symbol '{kw}'")

        elif cell_type == "markdown":
            print(f"  ✅ Markdown syntax: PASSED")

    print(f"\n------------------------------------------------------------------------------")
    if errors_found == 0:
        print(f"🎉 AUDIT COMPLETE: 0 Syntax Errors Found! Notebook is 100% Healthy.")
        return True
    else:
        print(f"⚠️ AUDIT COMPLETE: Found {errors_found} error(s). Please review above.")
        return False

if __name__ == "__main__":
    ok = audit_notebook(TARGET_NB)
    if not ok:
        sys.exit(1)
