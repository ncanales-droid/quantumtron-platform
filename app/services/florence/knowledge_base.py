"""
Knowledge Base System for Florence - Versión simplificada sin dependencias pesadas
"""

import os
import json
from typing import List, Dict, Optional, Any
from pathlib import Path
import hashlib
from datetime import datetime
import re


class ResearchKnowledgeBase:
    """Base de conocimiento simplificada para investigación académica"""
    
    def __init__(self, knowledge_dir: str = "data/knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        
        self.documents = []
        self.metadata = {}
        
        # Cargar documentos existentes
        self.load_documents()
        
        print(f"📚 Knowledge Base inicializada: {len(self.documents)} documentos")
    
    def add_document(self, text: str, metadata: Optional[Dict] = None) -> str:
        """Añadir documento a la base de conocimiento"""
        # Limpiar y normalizar texto
        text = self._clean_text(text)
        
        # Generar ID único
        doc_id = hashlib.md5(text.encode()).hexdigest()[:16]
        
        # Extraer metadatos automáticos
        auto_metadata = self._extract_metadata(text)
        
        # Combinar metadatos
        final_metadata = {
            "source": "user_input",
            "type": "research",
            "added": datetime.now().isoformat(),
            "length_chars": len(text),
            "length_words": len(text.split()),
            **auto_metadata,
            **(metadata or {})
        }
        
        document = {
            "id": doc_id,
            "text": text,
            "metadata": final_metadata,
            "added_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat()
        }
        
        self.documents.append(document)
        self._save_documents()
        
        print(f"✅ Documento añadido: {doc_id} ({len(text)} caracteres)")
        return doc_id
    
    def _clean_text(self, text: str) -> str:
        """Limpiar texto"""
        # Eliminar espacios múltiples y saltos de línea extras
        text = re.sub(r'\s+', ' ', text)
        # Eliminar caracteres no imprimibles
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t\r')
        return text.strip()
    
    def _extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extraer metadatos automáticamente del texto"""
        metadata = {}
        
        # Intentar detectar tipo de documento
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ['abstract', 'introduction', 'method', 'results', 'discussion', 'conclusion']):
            metadata['detected_type'] = 'academic_paper'
        
        # Contar párrafos
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        metadata['paragraphs'] = len(paragraphs)
        
        # Detectar posibles citas
        citation_patterns = [
            r'\(\w+ et al\.?, \d{4}\)',
            r'\(\w+, \d{4}\)',
            r'\b\d{4}\b.*\b\d{4}\b'
        ]
        
        citations = []
        for pattern in citation_patterns:
            citations.extend(re.findall(pattern, text))
        
        if citations:
            metadata['citation_count'] = len(set(citations))
        
        # Detectar estadísticas
        stat_patterns = [
            r'p\s*[=<>]\s*[\d\.]+',
            r't\(\d+\)\s*=\s*[\d\.]+',
            r'F\(\d+,\s*\d+\)\s*=\s*[\d\.]+',
            r'χ²\(\d+\)\s*=\s*[\d\.]+',
            r'r\s*=\s*[\d\.\-]+',
            r'd\s*=\s*[\d\.\-]+'
        ]
        
        stats = []
        for pattern in stat_patterns:
            stats.extend(re.findall(pattern, text_lower))
        
        if stats:
            metadata['statistical_terms'] = len(set(stats))
        
        return metadata
    
    def add_document_from_file(self, file_path: str, metadata: Optional[Dict] = None) -> str:
        """Añadir documento desde archivo"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            if not metadata:
                file_meta = {
                    "source": f"file:{Path(file_path).name}",
                    "file_type": Path(file_path).suffix,
                    "file_size": os.path.getsize(file_path)
                }
                metadata = file_meta
            else:
                metadata["source"] = f"file:{Path(file_path).name}"
            
            return self.add_document(text, metadata)
            
        except Exception as e:
            print(f"❌ Error cargando archivo {file_path}: {e}")
            return ""
    
    def search(self, query: str, top_k: int = 5, min_score: float = 0.1) -> List[Dict]:
        """Buscar documentos relevantes usando búsqueda semántica simple"""
        if not self.documents:
            return []
        
        query_terms = self._extract_search_terms(query)
        results = []
        
        for doc in self.documents:
            score = self._calculate_relevance_score(doc["text"], query_terms)
            
            if score >= min_score:
                # Extraer fragmento relevante
                snippet = self._extract_relevant_snippet(doc["text"], query_terms)
                
                results.append({
                    "id": doc["id"],
                    "text": snippet,
                    "full_text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": round(score, 3),
                    "search_method": "semantic_keyword"
                })
        
        # Ordenar por score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Actualizar último acceso
        for result in results[:top_k]:
            for doc in self.documents:
                if doc["id"] == result["id"]:
                    doc["last_accessed"] = datetime.now().isoformat()
        
        return results[:top_k]
    
    def _extract_search_terms(self, query: str) -> List[str]:
        """Extraer términos de búsqueda relevantes"""
        # Eliminar stopwords simples
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Tokenizar y limpiar
        terms = re.findall(r'\b\w+\b', query.lower())
        terms = [term for term in terms if term not in stopwords and len(term) > 2]
        
        # Agrupar términos compuestos (2-3 palabras)
        if len(terms) >= 2:
            bigrams = [f"{terms[i]} {terms[i+1]}" for i in range(len(terms)-1)]
            terms.extend(bigrams)
        
        return list(set(terms))
    
    def _calculate_relevance_score(self, text: str, query_terms: List[str]) -> float:
        """Calcular score de relevancia"""
        text_lower = text.lower()
        score = 0.0
        
        for term in query_terms:
            if ' ' in term:  # Es un bigrama
                if term in text_lower:
                    score += 0.3  # Bonus por coincidencia exacta de bigrama
            else:  # Palabra individual
                count = text_lower.count(term)
                if count > 0:
                    score += 0.1 * min(count, 5)  # Máximo 0.5 por término
        
        # Bonus por densidad de términos
        total_terms = len(query_terms)
        if total_terms > 0:
            terms_found = sum(1 for term in query_terms if term in text_lower)
            coverage = terms_found / total_terms
            score += coverage * 0.2
        
        # Normalizar a 0-1
        return min(score, 1.0)
    
    def _extract_relevant_snippet(self, text: str, query_terms: List[str], 
                                 snippet_length: int = 300) -> str:
        """Extraer fragmento relevante del texto"""
        if len(text) <= snippet_length:
            return text
        
        # Encontrar la posición con más términos de búsqueda
        best_position = 0
        best_score = -1
        
        for i in range(0, len(text) - snippet_length, snippet_length // 2):
            snippet = text[i:i + snippet_length]
            score = self._calculate_relevance_score(snippet, query_terms)
            
            if score > best_score:
                best_score = score
                best_position = i
        
        # Extraer snippet
        start = max(0, best_position - 50)
        end = min(len(text), start + snippet_length)
        
        snippet = text[start:end]
        
        # Añadir elipsis si es necesario
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        
        return snippet
    
    def query_with_context(self, question: str, context_chars: int = 2000) -> Dict:
        """Consultar con contexto relevante"""
        results = self.search(question, top_k=3, min_score=0.2)
        
        if not results:
            return {
                "context": "",
                "documents": [],
                "has_results": False,
                "message": "No relevant documents found"
            }
        
        # Construir contexto combinado
        context_parts = []
        total_chars = 0
        
        for i, result in enumerate(results):
            if total_chars >= context_chars:
                break
            
            available_chars = context_chars // len(results)
            excerpt = result["text"][:available_chars]
            
            context_parts.append(f"[Document {i+1} - Relevance: {result['score']:.2f}]:\n{excerpt}\n")
            total_chars += len(excerpt)
        
        return {
            "context": "\n".join(context_parts),
            "documents": results,
            "has_results": True,
            "total_documents": len(results),
            "average_relevance": sum(r["score"] for r in results) / len(results)
        }
    
    def _save_documents(self):
        """Guardar documentos en disco"""
        try:
            save_path = self.knowledge_dir / "documents.json"
            
            save_data = {
                "documents": self.documents,
                "metadata": self.metadata,
                "stats": self.get_statistics(),
                "last_updated": datetime.now().isoformat(),
                "version": "2.0"
            }
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"⚠️  Error guardando documentos: {e}")
    
    def load_documents(self):
        """Cargar documentos desde disco"""
        json_path = self.knowledge_dir / "documents.json"
        
        try:
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])
                    self.metadata = data.get("metadata", {})
                    
                    print(f"📚 Knowledge Base cargada: {len(self.documents)} documentos")
                    return True
                    
        except Exception as e:
            print(f"⚠️  Error cargando Knowledge Base: {e}")
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de la base de conocimiento"""
        if not self.documents:
            return {
                "total_documents": 0,
                "message": "Knowledge Base is empty"
            }
        
        total_chars = sum(len(doc["text"]) for doc in self.documents)
        total_words = sum(len(doc["text"].split()) for doc in self.documents)
        
        # Analizar tipos de documentos
        doc_types = {}
        for doc in self.documents:
            doc_type = doc["metadata"].get("detected_type", "unknown")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        # Calcular antigüedad
        now = datetime.now()
        ages = []
        for doc in self.documents:
            added = datetime.fromisoformat(doc["added_at"])
            age_days = (now - added).days
            ages.append(age_days)
        
        avg_age = sum(ages) / len(ages) if ages else 0
        
        return {
            "total_documents": len(self.documents),
            "total_characters": total_chars,
            "total_words": total_words,
            "average_document_length": total_chars / len(self.documents) if self.documents else 0,
            "document_types": doc_types,
            "average_age_days": round(avg_age, 1),
            "last_updated": self.metadata.get("last_updated", "never"),
            "storage_location": str(self.knowledge_dir.absolute())
        }
    
    def clear(self, confirm: bool = False) -> bool:
        """Limpiar toda la base de conocimiento"""
        if not confirm:
            print("⚠️  Para limpiar la base de conocimiento, llame a clear(confirm=True)")
            return False
        
        self.documents = []
        self.metadata = {}
        
        # Eliminar archivo de datos
        json_path = self.knowledge_dir / "documents.json"
        if json_path.exists():
            json_path.unlink()
        
        print("🧹 Knowledge Base limpiada completamente")
        return True
    
    def export_to_file(self, filename: str = "knowledge_base_export.json"):
        """Exportar base de conocimiento completa a archivo"""
        export_data = {
            "documents": self.documents,
            "metadata": self.metadata,
            "statistics": self.get_statistics(),
            "exported_at": datetime.now().isoformat(),
            "version": "2.0"
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📤 Knowledge Base exportada a: {filename}")
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Obtener documento por ID"""
        for doc in self.documents:
            if doc["id"] == doc_id:
                # Actualizar último acceso
                doc["last_accessed"] = datetime.now().isoformat()
                return doc
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Eliminar documento por ID"""
        for i, doc in enumerate(self.documents):
            if doc["id"] == doc_id:
                self.documents.pop(i)
                self._save_documents()
                print(f"🗑️  Documento eliminado: {doc_id}")
                return True
        return False


# Instancia global
knowledge_base = ResearchKnowledgeBase()
