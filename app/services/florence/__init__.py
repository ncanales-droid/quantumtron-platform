"""
Florence - PhD Research Agent
Advanced statistical analysis and research assistance
"""

__version__ = "1.0.0"
__author__ = "QuantumTron AI Team"

print(f"🔧 Inicializando Florence v{__version__}...")

# Core components - importar con manejo de errores robusto
try:
    from .deepseek_client import DeepSeekClient
    print("   ✅ DeepSeekClient importado")
    # Verificar si ya hay una instancia o crear una
    deepseek_client = DeepSeekClient()
    print("   ✅ DeepSeekClient instanciado")
except ImportError as e:
    print(f"   ❌ Error importando DeepSeekClient: {e}")
    DeepSeekClient = None
    deepseek_client = None
except Exception as e:
    print(f"   ⚠️  Error instanciando DeepSeekClient: {e}")
    deepseek_client = None

try:
    from .orchestrator import FlorenceOrchestrator
    print("   ✅ FlorenceOrchestrator importado")
except ImportError as e:
    print(f"   ❌ Error importando FlorenceOrchestrator: {e}")
    FlorenceOrchestrator = None

try:
    from .prompt_templates import (
        FLORENCE_SYSTEM_PROMPT,
        create_analysis_prompt,
        STATISTICAL_TEMPLATES,
        ACADEMIC_TEMPLATES
    )
    print("   ✅ Prompt templates importados")
except ImportError as e:
    print(f"   ⚠️  Error importando prompt templates: {e}")
    # Crear placeholders
    FLORENCE_SYSTEM_PROMPT = "You are Florence, a PhD research assistant."
    def create_analysis_prompt(question): return f"Analyze: {question}"
    STATISTICAL_TEMPLATES = {}
    ACADEMIC_TEMPLATES = {}

try:
    from .knowledge_base import ResearchKnowledgeBase, knowledge_base
    print("   ✅ Knowledge Base importada (versión simplificada)")
except ImportError as e:
    print(f"   ❌ Error importando Knowledge Base: {e}")
    ResearchKnowledgeBase = None
    knowledge_base = None

try:
    from .statistical_engine import StatisticalPhDEngine, statistical_engine
    print("   ✅ Statistical Engine importado")
except ImportError as e:
    print(f"   ❌ Error importando Statistical Engine: {e}")
    StatisticalPhDEngine = None
    statistical_engine = None

try:
    from .response_formatter import ResearchResponseFormatter, response_formatter
    print("   ✅ Response Formatter importado")
except ImportError as e:
    print(f"   ❌ Error importando Response Formatter: {e}")
    ResearchResponseFormatter = None
    response_formatter = None

__all__ = [
    # Core
    "FlorenceOrchestrator",
    "DeepSeekClient",
    "deepseek_client",
    
    # Prompt templates
    "FLORENCE_SYSTEM_PROMPT",
    "create_analysis_prompt",
    "STATISTICAL_TEMPLATES",
    "ACADEMIC_TEMPLATES",
    
    # New components
    "ResearchKnowledgeBase",
    "knowledge_base",
    "StatisticalPhDEngine", 
    "statistical_engine",
    "ResearchResponseFormatter",
    "response_formatter"
]

print(f"\n🎯 Florence v{__version__} inicializado:")
print(f"   🤖 DeepSeek Client: {'✅ Ready' if deepseek_client else '❌ Not available'}")
print(f"   🧠 Orchestrator: {'✅ Ready' if FlorenceOrchestrator else '❌ Not available'}")
print(f"   📊 Knowledge Base: {'✅ Ready' if knowledge_base else '❌ Not available'}")
print(f"   📈 Statistical Engine: {'✅ Ready' if statistical_engine else '❌ Not available'}")
print(f"   📝 Response Formatter: {'✅ Ready' if response_formatter else '❌ Not available'}")
print(f"\n🚀 {'✅ Florence está listo para usar!' if deepseek_client else '⚠️  Florence tiene problemas de configuración'}")
