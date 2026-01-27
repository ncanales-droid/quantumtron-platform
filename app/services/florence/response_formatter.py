"""
Response Formatter for Florence - Academic output formatting
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import re


class OutputFormat(Enum):
    """Formatos de salida disponibles"""
    PLAIN_TEXT = "plain"
    MARKDOWN = "markdown"
    HTML = "html"
    LATEX = "latex"
    JSON = "json"
    PDF_READY = "pdf_ready"


class CitationStyle(Enum):
    """Estilos de citación académica"""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    HARVARD = "harvard"
    IEEE = "ieee"
    VANCOUVER = "vancouver"


class ResearchResponseFormatter:
    """Formateador de respuestas académicas para Florence"""
    
    def __init__(self, default_format: OutputFormat = OutputFormat.MARKDOWN,
                 citation_style: CitationStyle = CitationStyle.APA):
        self.default_format = default_format
        self.citation_style = citation_style
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict:
        """Cargar templates de formato"""
        return {
            "title": {
                "markdown": "# {title}\n\n",
                "html": "<h1>{title}</h1>\n",
                "latex": "\\section{{{title}}}\n",
                "plain": "{title}\n{underline}\n\n"
            },
            "section": {
                "markdown": "## {section}\n\n",
                "html": "<h2>{section}</h2>\n",
                "latex": "\\subsection{{{section}}}\n",
                "plain": "{section}\n{'-'*len(section)}\n\n"
            },
            "subsection": {
                "markdown": "### {subsection}\n\n",
                "html": "<h3>{subsection}</h3>\n",
                "latex": "\\subsubsection{{{subsection}}}\n",
                "plain": "{subsection}\n{'.'*len(subsection)}\n\n"
            },
            "paragraph": {
                "markdown": "{text}\n\n",
                "html": "<p>{text}</p>\n",
                "latex": "{text}\n\n",
                "plain": "{text}\n\n"
            },
            "list_item": {
                "markdown": "- {item}\n",
                "html": "<li>{item}</li>\n",
                "latex": "\\item {item}\n",
                "plain": "• {item}\n"
            },
            "code_block": {
                "markdown": "```{language}\n{code}\n```\n\n",
                "html": "<pre><code class='language-{language}'>{code}</code></pre>\n",
                "latex": "\\begin{{lstlisting}}[language={language}]\n{code}\n\\end{{lstlisting}}\n",
                "plain": "{code}\n\n"
            },
            "table": {
                "markdown": self._format_markdown_table,
                "html": self._format_html_table,
                "latex": self._format_latex_table,
                "plain": self._format_plain_table
            },
            "citation": {
                "apa": self._format_apa_citation,
                "mla": self._format_mla_citation,
                "chicago": self._format_chicago_citation
            }
        }
    
    def format_response(self, content: Dict[str, Any], 
                       output_format: Optional[OutputFormat] = None) -> str:
        """Formatear respuesta completa"""
        format_type = output_format or self.default_format
        format_str = format_type.value
        
        # Determinar tipo de contenido
        content_type = content.get("type", "analysis")
        
        if content_type == "statistical_report":
            return self._format_statistical_report(content, format_str)
        elif content_type == "research_methodology":
            return self._format_methodology(content, format_str)
        elif content_type == "literature_review":
            return self._format_literature_review(content, format_str)
        elif content_type == "data_analysis":
            return self._format_data_analysis(content, format_str)
        else:
            return self._format_general_analysis(content, format_str)
    
    def _format_statistical_report(self, content: Dict, fmt: str) -> str:
        """Formatear reporte estadístico"""
        sections = []
        
        # Título
        title = content.get("title", "Statistical Analysis Report")
        sections.append(self._apply_template("title", fmt, title=title))
        
        if fmt == "plain":
            sections[-1] = self._apply_template("title", fmt, 
                title=title, underline="="*len(title))
        
        # Metadatos
        metadata = content.get("metadata", {})
        if metadata:
            sections.append(self._apply_template("section", fmt, section="Metadata"))
            for key, value in metadata.items():
                sections.append(f"**{key}**: {value}\n" if fmt == "markdown" else f"{key}: {value}\n")
            sections.append("\n")
        
        # Resumen ejecutivo
        if "executive_summary" in content:
            sections.append(self._apply_template("section", fmt, section="Executive Summary"))
            sections.append(self._apply_template("paragraph", fmt, text=content["executive_summary"]))
        
        # Métodos
        if "methods" in content:
            sections.append(self._apply_template("section", fmt, section="Methods"))
            methods = content["methods"]
            
            if isinstance(methods, list):
                for method in methods:
                    sections.append(self._apply_template("list_item", fmt, item=method))
                sections.append("\n")
            else:
                sections.append(self._apply_template("paragraph", fmt, text=str(methods)))
        
        # Resultados
        if "results" in content:
            sections.append(self._apply_template("section", fmt, section="Results"))
            results = content["results"]
            
            if isinstance(results, dict):
                # Tabla de resultados estadísticos
                if "statistical_tests" in results:
                    sections.append(self._apply_template("subsection", fmt, 
                        subsection="Statistical Tests"))
                    
                    tests = results["statistical_tests"]
                    if isinstance(tests, list) and tests:
                        # Crear tabla
                        headers = ["Test", "Statistic", "p-value", "Effect Size", "Interpretation"]
                        rows = []
                        
                        for test in tests[:10]:  # Limitar a 10 tests
                            rows.append([
                                test.get("test", "N/A"),
                                f"{test.get('statistic', 0):.3f}",
                                f"{test.get('p_value', 1):.4f}",
                                f"{test.get('effect_size', 0):.3f}" if test.get('effect_size') else "N/A",
                                test.get('interpretation', '')[:50] + "..."
                            ])
                        
                        sections.append(self._apply_template("table", fmt,
                            headers=headers, rows=rows))
                
                # Descripción de datos
                if "descriptive_stats" in results:
                    sections.append(self._apply_template("subsection", fmt,
                        subsection="Descriptive Statistics"))
                    
                    stats = results["descriptive_stats"]
                    if isinstance(stats, dict):
                        # Convertir a tabla
                        headers = ["Variable", "Mean", "SD", "Min", "Median", "Max", "N"]
                        rows = []
                        
                        for var_name, var_stats in stats.items():
                            if isinstance(var_stats, dict) and var_stats.get("type") == "numeric":
                                rows.append([
                                    var_name,
                                    f"{var_stats.get('mean', 0):.2f}",
                                    f"{var_stats.get('std', 0):.2f}",
                                    f"{var_stats.get('min', 0):.2f}",
                                    f"{var_stats.get('median', 0):.2f}",
                                    f"{var_stats.get('max', 0):.2f}",
                                    str(var_stats.get('count', 0))
                                ])
                        
                        if rows:
                            sections.append(self._apply_template("table", fmt,
                                headers=headers, rows=rows))
        
        # Discusión
        if "discussion" in content:
            sections.append(self._apply_template("section", fmt, section="Discussion"))
            sections.append(self._apply_template("paragraph", fmt, text=content["discussion"]))
        
        # Conclusiones
        if "conclusions" in content:
            sections.append(self._apply_template("section", fmt, section="Conclusions"))
            conclusions = content["conclusions"]
            
            if isinstance(conclusions, list):
                for i, conclusion in enumerate(conclusions, 1):
                    sections.append(f"{i}. {conclusion}\n")
            else:
                sections.append(self._apply_template("paragraph", fmt, text=str(conclusions)))
        
        # Limitaciones
        if "limitations" in content:
            sections.append(self._apply_template("section", fmt, section="Limitations"))
            limitations = content["limitations"]
            
            if isinstance(limitations, list):
                for limitation in limitations:
                    sections.append(self._apply_template("list_item", fmt, item=limitation))
                sections.append("\n")
        
        # Recomendaciones
        if "recommendations" in content:
            sections.append(self._apply_template("section", fmt, section="Recommendations"))
            recommendations = content["recommendations"]
            
            if isinstance(recommendations, list):
                for rec in recommendations:
                    sections.append(self._apply_template("list_item", fmt, item=rec))
                sections.append("\n")
        
        # Anexos (código, datos, etc.)
        if "appendices" in content:
            sections.append(self._apply_template("section", fmt, section="Appendices"))
            appendices = content["appendices"]
            
            for appendix_name, appendix_content in appendices.items():
                sections.append(self._apply_template("subsection", fmt, 
                    subsection=f"Appendix: {appendix_name}"))
                
                if isinstance(appendix_content, str) and "```" in appendix_content:
                    # Es código
                    language = "python" if "def " in appendix_content or "import " in appendix_content else "text"
                    sections.append(self._apply_template("code_block", fmt,
                        language=language, code=appendix_content))
                else:
                    sections.append(self._apply_template("paragraph", fmt, 
                        text=str(appendix_content)))
        
        # Firma
        sections.append("\n---\n")
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sections.append(f"*Generated by Florence PhD Agent on {date_str}*\n")
        
        return "".join(sections)
    
    def _format_methodology(self, content: Dict, fmt: str) -> str:
        """Formatear sección de metodología"""
        # Implementación similar...
        return self._format_general_analysis(content, fmt)
    
    def _format_literature_review(self, content: Dict, fmt: str) -> str:
        """Formatear revisión de literatura"""
        # Implementación similar...
        return self._format_general_analysis(content, fmt)
    
    def _format_data_analysis(self, content: Dict, fmt: str) -> str:
        """Formatear análisis de datos"""
        # Implementación similar...
        return self._format_general_analysis(content, fmt)
    
    def _format_general_analysis(self, content: Dict, fmt: str) -> str:
        """Formatear análisis general"""
        output = []
        
        if "title" in content:
            title = content["title"]
            output.append(self._apply_template("title", fmt, title=title))
            
            if fmt == "plain":
                output[-1] = self._apply_template("title", fmt, 
                    title=title, underline="="*len(title))
        
        if "content" in content:
            if isinstance(content["content"], list):
                for item in content["content"]:
                    if isinstance(item, dict):
                        if item.get("type") == "paragraph":
                            output.append(self._apply_template("paragraph", fmt, 
                                text=item.get("text", "")))
                        elif item.get("type") == "list":
                            for list_item in item.get("items", []):
                                output.append(self._apply_template("list_item", fmt, 
                                    item=list_item))
                            output.append("\n")
                    else:
                        output.append(str(item) + "\n")
            else:
                output.append(self._apply_template("paragraph", fmt, 
                    text=str(content["content"])))
        
        return "".join(output)
    
    def _apply_template(self, template_type: str, fmt: str, **kwargs) -> str:
        """Aplicar template específico"""
        template = self.templates.get(template_type, {}).get(fmt)
        
        if callable(template):
            return template(**kwargs)
        elif template:
            return template.format(**kwargs)
        else:
            # Fallback a texto plano
            if "text" in kwargs:
                return kwargs["text"] + "\n"
            elif "title" in kwargs:
                return kwargs["title"] + "\n\n"
            return ""
    
    # Métodos para formatos de tabla
    def _format_markdown_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Formatear tabla en Markdown"""
        table = []
        
        # Headers
        table.append("| " + " | ".join(headers) + " |")
        table.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        # Rows
        for row in rows:
            table.append("| " + " | ".join(str(cell) for cell in row) + " |")
        
        table.append("")  # Línea vacía al final
        return "\n".join(table)
    
    def _format_html_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Formatear tabla en HTML"""
        table = ['<table class="statistical-table">']
        
        # Headers
        table.append("  <thead>")
        table.append("    <tr>")
        for header in headers:
            table.append(f"      <th>{header}</th>")
        table.append("    </tr>")
        table.append("  </thead>")
        
        # Body
        table.append("  <tbody>")
        for row in rows:
            table.append("    <tr>")
            for cell in row:
                table.append(f"      <td>{cell}</td>")
            table.append("    </tr>")
        table.append("  </tbody>")
        
        table.append("</table>")
        return "\n".join(table)
    
    def _format_latex_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Formatear tabla en LaTeX"""
        table = []
        table.append("\\begin{table}[htbp]")
        table.append("\\centering")
        table.append("\\begin{tabular}{|" + "c|" * len(headers) + "}")
        table.append("\\hline")
        
        # Headers
        table.append(" & ".join(headers) + " \\\\")
        table.append("\\hline")
        
        # Rows
        for row in rows:
            table.append(" & ".join(str(cell) for cell in row) + " \\\\")
            table.append("\\hline")
        
        table.append("\\end{tabular}")
        table.append("\\caption{Statistical Results}")
        table.append("\\label{tab:results}")
        table.append("\\end{table}")
        return "\n".join(table)
    
    def _format_plain_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Formatear tabla en texto plano"""
        # Calcular anchos de columna
        col_widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Crear tabla
        table = []
        
        # Headers
        header_line = "  ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
        table.append(header_line)
        table.append("-" * len(header_line))
        
        # Rows
        for row in rows:
            table.append("  ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)))
        
        table.append("")  # Línea vacía al final
        return "\n".join(table)
    
    # Métodos para citaciones
    def _format_apa_citation(self, author: str, year: int, title: str, 
                            journal: str = "", volume: str = "", pages: str = "") -> str:
        """Formatear cita APA"""
        citation = f"{author} ({year}). {title}."
        if journal:
            citation += f" *{journal}*"
            if volume:
                citation += f", {volume}"
            if pages:
                citation += f", {pages}"
        citation += "."
        return citation
    
    def _format_mla_citation(self, author: str, title: str, journal: str = "",
                            year: int = None, pages: str = "") -> str:
        """Formatear cita MLA"""
        citation = f"{author}. \"{title}.\""
        if journal:
            citation += f" *{journal}*"
        if year:
            citation += f", {year}"
        if pages:
            citation += f", pp. {pages}."
        else:
            citation += "."
        return citation
    
    def _format_chicago_citation(self, author: str, title: str, journal: str = "",
                                year: int = None, volume: str = "", pages: str = "") -> str:
        """Formatear cita Chicago"""
        citation = f"{author}. \"{title}.\""
        if journal:
            citation += f" *{journal}*"
            if volume:
                citation += f" {volume}"
        if year:
            citation += f" ({year})"
        if pages:
            citation += f": {pages}."
        else:
            citation += "."
        return citation
    
    def format_citation(self, citation_data: Dict) -> str:
        """Formatear cita según estilo configurado"""
        style_method = self.templates["citation"].get(self.citation_style.value)
        if style_method:
            return style_method(**citation_data)
        return f"{citation_data.get('author', 'Unknown')} ({citation_data.get('year', 'n.d.')}). {citation_data.get('title', 'Untitled')}."
    
    def export_to_file(self, content: Dict, filename: str, 
                      format_type: Optional[OutputFormat] = None):
        """Exportar contenido a archivo"""
        formatted = self.format_response(content, format_type)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(formatted)
        
        print(f"✅ Exportado a: {filename} ({len(formatted)} caracteres)")


# Instancia global
response_formatter = ResearchResponseFormatter()
