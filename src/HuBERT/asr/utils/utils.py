# utils.py (reemplazar la función actual)
import unicodedata
import re

def normalize_text(text: str) -> str:
    """
    Normalización robusta y multilingüe.
    - Normaliza Unicode (NFKC)
    - Aplica casefold (case-insensitive para scripts que tienen case)
    - Elimina caracteres que no sean letras o números (según categoría Unicode)
    - Normaliza espacios a un único espacio
    - Si el resultado queda vacío, retorna "FORBIDDEN_SENTENCE"
    """
    if text is None:
        return "FORBIDDEN_SENTENCE"

    # NFKC para normalizar la representación Unicode
    text = unicodedata.normalize("NFKC", str(text).strip())

    # Casefold es más agresivo y correcto para Unicode que lower()
    text = text.casefold()

    out_chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        # Mantener letras (L*), números (N*), y espacios (Zs -> we will map to ' ')
        if cat.startswith("L") or cat.startswith("N"):
            out_chars.append(ch)
        elif cat == "Zs":
            out_chars.append(" ")
        else:
            # eliminar puntuación, símbolos, control chars, etc.
            # si quieres permitir algún signo concreto, añade una condición aquí
            continue

    # Reemplazar múltiples espacios por uno, y strip
    out = re.sub(r"\s+", " ", "".join(out_chars)).strip()

    if not out:
        return "FORBIDDEN_SENTENCE"
    return out
