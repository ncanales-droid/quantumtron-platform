"""
Orquestador principal para Florence.
Coordina todos los componentes del agente.
"""
import logging
from typing import Dict, Any
from app.services.florence.deepseek_client import DeepSeekClient
from app.services.florence.prompt_templates import (
    FLORENCE_SYSTEM_PROMPT, 
    create_analysis_prompt
)

logger = logging.getLogger(__name__)

class FlorenceOrchestrator:
    """Orquestador principal del agente Florence."""
    
    def __init__(self):
        self.deepseek = DeepSeekClient()
        logger.info("Florence PhD Statistician Agent initialized")
    
    async def analyze_dataset(
        self, 
        data_summary: Dict[str, Any], 
        user_question: str
    ) -> Dict[str, Any]:
        """
        Analizar dataset con Florence.
        
        Args:
            data_summary: Resumen estadístico del dataset
            user_question: Pregunta del usuario
            
        Returns:
            Análisis completo de Florence
        """
        try:
            # 1. Formatear resumen de datos
            formatted_summary = self._format_data_summary(data_summary)
            
            # 2. Crear mensajes para DeepSeek
            messages = [
                {"role": "system", "content": FLORENCE_SYSTEM_PROMPT},
                {"role": "user", "content": create_analysis_prompt(formatted_summary, user_question)}
            ]
            
            # 3. Llamar a DeepSeek
            logger.info(f"Florence analyzing data for question: {user_question}")
            response = await self.deepseek.chat_completion(messages, temperature=0.7)
            
            # 4. Formatear respuesta
            result = {
                "analysis": response["content"],
                "metadata": {
                    "model": response["model"],
                    "tokens_used": response["tokens_used"],
                    "agent": "Florence PhD Statistician",
                    "version": "1.0.0"
                }
            }
            
            logger.info(f"Florence analysis complete. Tokens used: {response['tokens_used']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in Florence analysis: {str(e)}")
            raise
    
    def _format_data_summary(self, data_summary: Dict[str, Any]) -> str:
        """Formatear resumen de datos para el prompt."""
        lines = []
        
        if "rows" in data_summary and "columns" in data_summary:
            lines.append(f"Dataset: {data_summary['rows']} rows × {data_summary['columns']} columns")
        
        if "column_names" in data_summary:
            lines.append(f"Variables: {', '.join(data_summary['column_names'][:10])}")
            if len(data_summary['column_names']) > 10:
                lines.append(f"... and {len(data_summary['column_names']) - 10} more variables")
        
        if "data_types" in data_summary:
            lines.append("\nData Types:")
            for col, dtype in list(data_summary['data_types'].items())[:5]:
                lines.append(f"  - {col}: {dtype}")
        
        if "summary" in data_summary:
            lines.append(f"\nSummary: {data_summary['summary']}")
        
        return "\n".join(lines)
