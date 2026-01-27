"""
Templates de prompts para Florence, el PhD Statistician.
"""

FLORENCE_SYSTEM_PROMPT = """
# IDENTITY: Dr. Florence, PhD in Statistics
# EXPERTISE: 20+ years in statistical consulting
# AFFILIATION: Professor Emerita, Stanford Statistics Department
# LANGUAGES: Fluent in English and Spanish

## CORE PRINCIPLES:
1. **Statistical Rigor**: Always recommend methodologically sound approaches
2. **Practical Relevance**: Connect statistical findings to real-world implications
3. **Educational Focus**: Explain concepts clearly with examples
4. **Ethical Transparency**: Acknowledge limitations and assumptions
5. **Language Flexibility**: Respond in the same language as the user's question

## COMMUNICATION STYLE:
- Professional yet approachable
- Use precise statistical terminology but explain it
- Provide actionable recommendations
- Reference academic literature when appropriate
- Include practical implementation guidance
- Match the user's language (English/Spanish)

## RESPONSE STRUCTURE:
1. **Executive Summary** (2-3 sentences)
2. **Methodological Approach** (appropriate tests/methods)
3. **Statistical Findings** (interpret results)
4. **Practical Implications** (what it means in context)
5. **Recommendations** (next steps)
6. **References** (key academic sources)
7. **Code Examples** (Python/R if applicable)

## KEY KNOWLEDGE AREAS:
- Experimental Design & ANOVA
- Regression Modeling
- Bayesian Inference
- Machine Learning Evaluation
- Time Series Analysis
- Multivariate Methods

## LANGUAGE POLICY:
- Detect user's language from their question
- Respond in the same language
- Maintain statistical terminology consistency
- Translate concepts when helpful

Always ask clarifying questions if data or context is insufficient.
"""

def create_analysis_prompt(data_summary: str, user_question: str) -> str:
    """
    Crear prompt para análisis estadístico.
    
    Args:
        data_summary: Resumen estadístico del dataset
        user_question: Pregunta del usuario
        
    Returns:
        Prompt formateado
    """
    return f"""
## DATASET FOR ANALYSIS
{data_summary}

## USER'S QUESTION
{user_question}

## LANGUAGE INSTRUCTION:
Respond in the same language as the user's question.

## ANALYSIS INSTRUCTIONS:
As Dr. Florence, provide a comprehensive statistical analysis.
Focus on methodological soundness and practical applicability.
"""
# Agregar al final de prompt_templates.py
STATISTICAL_TEMPLATES = {
    "t_test": """
    Analyze the following t-test results:
    t({df}) = {t_value}, p = {p_value}
    
    Provide interpretation in academic format including:
    1. Statistical significance
    2. Effect size (if available)
    3. Practical implications
    4. Limitations
    """,
    
    "anova": """
    Analyze the following ANOVA results:
    F({df_between}, {df_within}) = {f_value}, p = {p_value}
    
    Provide interpretation including:
    1. Overall significance
    2. Effect size (eta-squared)
    3. Post-hoc recommendations
    4. Assumptions check
    """,
    
    "correlation": """
    Analyze the correlation: r = {r_value}, p = {p_value}, n = {n}
    
    Interpretation should include:
    1. Strength and direction
    2. Statistical significance
    3. Effect size interpretation
    4. Causation warnings
    """,
    
    "regression": """
    Analyze regression model: R² = {r_squared}, F = {f_value}, p = {f_p}
    
    For coefficients:
    {coefficients_table}
    
    Provide analysis including:
    1. Model fit
    2. Significant predictors
    3. Coefficient interpretation
    4. Model assumptions
    """
}

ACADEMIC_TEMPLATES = {
    "abstract": """
    Write an academic abstract for a study with:
    Objective: {objective}
    Methods: {methods}
    Results: {results}
    Conclusion: {conclusion}
    
    Format: 150-250 words, structured abstract.
    """,
    
    "introduction": """
    Write an introduction section covering:
    Background: {background}
    Problem statement: {problem}
    Research question: {question}
    Hypothesis: {hypothesis}
    
    Include relevant citations and theoretical framework.
    """,
    
    "methodology": """
    Describe research methodology:
    Design: {design}
    Participants: {participants}
    Measures: {measures}
    Procedure: {procedure}
    Analysis: {analysis}
    
    Follow APA/ICMJE guidelines.
    """,
    
    "discussion": """
    Write discussion section:
    Main findings: {findings}
    Comparison with literature: {comparison}
    Implications: {implications}
    Limitations: {limitations}
    Future research: {future}
    
    Interpret results without repeating them.
    """
}
