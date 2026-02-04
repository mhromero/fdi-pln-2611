import json
from typing import Any, Dict

from .config import GOLD_RESOURCE_NAME, PROMPT_BASE
from .ollama_client import ollama

def analizar_oferta(
    oferta: Dict[str, Any],
    needs: Dict[str, Any],
    surplus: Dict[str, int],
) -> Dict[str, Any]:
    """
    Usa Ollama para interpretar una carta y devolver un JSON con
    tipo (oferta|confirmacion|otro), oferta, pide, recursos_recibidos.
    """
    prompt = f"""
Eres un asistente que toma decisiones sobre ofertas recibidas.

Tu tarea es LEER la oferta y devolver un JSON estructurado con esta forma:

{{
  "decision": "aceptada" | "rechazada",
  "oferta": {{
    "recurso": cantidad entero
  }},
  "pide": {{
    "recurso": cantidad entero
  }},
}}

Donde:
- "decision" = "aceptada" si se cumplen TODAS las condiciones siguientes:
    a) los recursos que ofrece los necesitamos para cumplir el objetivo
    b) no pide recursos que necesitemos para nuestro objetivo
    c) podemos dar los recursos que pide
    d) se envían como máximo el mismo número de recursos que se reciben, salvo que con la oferta completemos el objetivo al 100%

- "decision" = "rechazada" si no se cumple alguna de las condiciones anteriores
- "oferta" describe lo que EL OTRO agente nos ofrece.
- "pide" describe lo que EL OTRO agente quiere que le enviemos.

IMPORTANTE:
- Devuelve SIEMPRE un JSON VÁLIDO, sin texto adicional.

NECESITAMOS:
{json.dumps(needs, ensure_ascii=False, indent=2)}

PODEMOS OFRECER:
{json.dumps(surplus, ensure_ascii=False, indent=2)}

OFERTA A ANALIZAR:
{json.dumps(oferta, ensure_ascii=False, indent=2)}


"""
    
    respuesta = ollama(prompt)
    
    try:
        data = json.loads(respuesta)
        if not isinstance(data, dict):
            raise ValueError("Respuesta no es un dict")
        return data
    except (json.JSONDecodeError, ValueError):
        print("ERROR: Ollama no devolvió JSON válido al analizar carta")
        print(respuesta)
        return {"tipo": "otro", "oferta": {}, "pide": {}, "recursos_recibidos": {}}


def generar_plan(info: Dict[str, Any], people: list, cartas_recibidas: list) -> Dict[str, Any] | None:
    """
    Función legacy: genera un plan (acciones + análisis) con Ollama.
    Se deja por compatibilidad o reutilización del prompt.
    """
    prompt = f"""
NUESTROS RECURSOS:
{json.dumps(info, indent=2)}

AGENTES DISPONIBLES:
{json.dumps(people, indent=2)}

CARTAS RECIBIDAS:
{json.dumps(cartas_recibidas, indent=2)}
""" + PROMPT_BASE

    respuesta = ollama(prompt)
    try:
        return json.loads(respuesta)
    except json.JSONDecodeError:
        print("ERROR: Ollama no devolvió JSON válido")
        print(respuesta)
        return None
