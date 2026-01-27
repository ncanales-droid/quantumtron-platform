"""
Statistical Engine for Florence - Advanced statistical analysis - FIXED VERSION
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import json
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    import statsmodels.stats.api as sms
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("⚠️  statsmodels no instalado. Para análisis avanzado: pip install statsmodels")

try:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  scikit-learn no instalado. Para ML: pip install scikit-learn")


@dataclass
class StatisticalResult:
    """Resultado de análisis estadístico"""
    test_name: str
    statistic: float
    p_value: float
    df: Optional[Tuple] = None
    effect_size: Optional[float] = None
    ci_95: Optional[Tuple[float, float]] = None
    interpretation: str = ""
    assumptions: Dict[str, bool] = None
    recommendations: List[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "test": self.test_name,
            "statistic": float(self.statistic) if hasattr(self.statistic, "__float__") else self.statistic,
            "p_value": float(self.p_value),
            "degrees_of_freedom": self.df,
            "effect_size": float(self.effect_size) if self.effect_size else None,
            "confidence_interval": self.ci_95,
            "interpretation": self.interpretation,
            "assumptions_met": self.assumptions or {},
            "recommendations": self.recommendations or []
        }


class StatisticalPhDEngine:
    """Motor estadístico avanzado para investigación PhD - VERSIÓN CORREGIDA"""
    
    def __init__(self):
        self.results_history = []
        
    def describe_data(self, data: Union[pd.DataFrame, np.ndarray, List], 
                     variable_names: Optional[List[str]] = None) -> Dict:
        """Análisis descriptivo completo - VERSIÓN CORREGIDA"""
        # CORRECCIÓN: Manejar lista de arrays de diferente longitud
        if isinstance(data, list):
            # Verificar si todos los elementos son arrays numpy
            if all(isinstance(item, np.ndarray) for item in data):
                # Si son arrays 1D, crear DataFrame con diferentes columnas
                if all(item.ndim == 1 for item in data):
                    # Para lista de arrays 1D (como [group1, group2])
                    return self._describe_list_of_arrays(data, variable_names)
                else:
                    # Arrays multidimensionales - intentar convertirlos
                    try:
                        data_array = np.array(data)
                        if data_array.ndim == 1:
                            df = pd.DataFrame(data_array, columns=variable_names or ["Variable"])
                        else:
                            df = pd.DataFrame(data_array, columns=variable_names or [f"Var_{i}" for i in range(data_array.shape[1])])
                    except:
                        # Si falla, tratar cada elemento por separado
                        return self._describe_list_of_arrays(data, variable_names)
            else:
                # Lista de otros tipos
                try:
                    data_array = np.array(data)
                    df = pd.DataFrame(data_array, columns=variable_names or [f"Var_{i}" for i in range(data_array.shape[1])])
                except:
                    # Último recurso
                    df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                df = pd.DataFrame(data, columns=variable_names or ["Variable"])
            else:
                df = pd.DataFrame(data, columns=variable_names or [f"Var_{i}" for i in range(data.shape[1])])
        else:
            raise ValueError(f"Tipo de datos no soportado: {type(data)}")
        
        description = {}
        
        for col in df.columns:
            col_data = df[col].dropna()
            
            if pd.api.types.is_numeric_dtype(col_data):
                # Estadísticas para variables numéricas
                description[col] = {
                    "type": "numeric",
                    "count": int(len(col_data)),
                    "missing": int(df[col].isna().sum()),
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "25%": float(col_data.quantile(0.25)),
                    "median": float(col_data.median()),
                    "75%": float(col_data.quantile(0.75)),
                    "max": float(col_data.max()),
                    "skewness": float(col_data.skew()),
                    "kurtosis": float(col_data.kurtosis()),
                    "shapiro_p": float(stats.shapiro(col_data)[1]) if len(col_data) >= 3 and len(col_data) <= 5000 else None
                }
            else:
                # Estadísticas para variables categóricas
                value_counts = col_data.value_counts()
                description[col] = {
                    "type": "categorical",
                    "count": int(len(col_data)),
                    "missing": int(df[col].isna().sum()),
                    "unique_values": int(col_data.nunique()),
                    "mode": str(col_data.mode()[0]) if not col_data.mode().empty else None,
                    "top_categories": value_counts.head(5).to_dict()
                }
        
        return {
            "dataset_info": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "total_missing": df.isna().sum().sum(),
                "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"
            },
            "variables": description
        }
    
    def _describe_array(self, arr: np.ndarray, name: str) -> Dict:
        """Describir un array individual"""
        arr_clean = arr[~np.isnan(arr)]
        
        if len(arr_clean) == 0:
            return {name: {"type": "numeric", "count": 0, "missing": len(arr), "note": "All values are NaN"}}
        
        return {
            name: {
                "type": "numeric",
                "count": int(len(arr_clean)),
                "missing": int(len(arr) - len(arr_clean)),
                "mean": float(np.mean(arr_clean)),
                "std": float(np.std(arr_clean, ddof=1)),
                "min": float(np.min(arr_clean)),
                "25%": float(np.percentile(arr_clean, 25)),
                "median": float(np.median(arr_clean)),
                "75%": float(np.percentile(arr_clean, 75)),
                "max": float(np.max(arr_clean)),
                "skewness": float(stats.skew(arr_clean)),
                "kurtosis": float(stats.kurtosis(arr_clean)),
                "shapiro_p": float(stats.shapiro(arr_clean)[1]) if 3 <= len(arr_clean) <= 5000 else None
            }
        }
    
    def _describe_list_of_arrays(self, arrays: List[np.ndarray], variable_names: Optional[List[str]] = None) -> Dict:
        """Describir lista de arrays (grupos)"""
        description = {}
        
        for i, arr in enumerate(arrays):
            name = variable_names[i] if variable_names and i < len(variable_names) else f"Group_{i+1}"
            description.update(self._describe_array(arr, name))
        
        return {
            "dataset_info": {
                "total_groups": len(arrays),
                "total_samples": sum(len(arr) for arr in arrays),
                "note": "Each group analyzed separately"
            },
            "variables": description
        }
    
    def t_test(self, group1: np.ndarray, group2: np.ndarray, 
               test_type: str = "independent", alternative: str = "two-sided") -> StatisticalResult:
        """Prueba t para comparación de medias"""
        # [Mantener el resto del método igual...]
        # Verificar supuestos
        assumptions = {
            "normality": len(group1) >= 3 and len(group2) >= 3,
            "equal_variance": True  # Usaremos Welch si no se cumple
        }
        
        # Test de normalidad (Shapiro-Wilk)
        if len(group1) >= 3 and len(group1) <= 5000:
            _, p1 = stats.shapiro(group1)
            assumptions["group1_normal"] = p1 > 0.05
        if len(group2) >= 3 and len(group2) <= 5000:
            _, p2 = stats.shapiro(group2)
            assumptions["group2_normal"] = p2 > 0.05
        
        # Test de homogeneidad de varianzas (Levene)
        if len(group1) > 1 and len(group2) > 1:
            _, p_levene = stats.levene(group1, group2)
            assumptions["equal_variance"] = p_levene > 0.05
        
        # Realizar prueba t
        if test_type == "independent":
            if assumptions["equal_variance"]:
                # t-test de Student
                t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=True, alternative=alternative)
                test_name = "Student's t-test (independent)"
            else:
                # t-test de Welch
                t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False, alternative=alternative)
                test_name = "Welch's t-test (independent)"
            df = (len(group1) - 1) + (len(group2) - 1)
        else:  # paired
            if len(group1) != len(group2):
                raise ValueError("Para t-test pareado, los grupos deben tener el mismo tamaño")
            t_stat, p_val = stats.ttest_rel(group1, group2, alternative=alternative)
            test_name = "Paired t-test"
            df = len(group1) - 1
        
        # Calcular tamaño del efecto (Cohen's d)
        pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std != 0 else 0
        
        # Calcular intervalo de confianza 95%
        if test_type == "independent":
            ci = stats.t.interval(0.95, df, loc=np.mean(group1)-np.mean(group2), 
                                 scale=pooled_std*np.sqrt(1/len(group1) + 1/len(group2)))
        else:
            diff = group1 - group2
            ci = stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), 
                                 scale=stats.sem(diff))
        
        # Interpretación
        if p_val < 0.001:
            significance = "altamente significativa"
        elif p_val < 0.01:
            significance = "muy significativa"
        elif p_val < 0.05:
            significance = "significativa"
        else:
            significance = "no significativa"
        
        interpretation = (
            f"La diferencia entre las medias es {significance} (p = {p_val:.4f}). "
            f"El tamaño del efecto (d = {cohens_d:.3f}) es {'grande' if abs(cohens_d) > 0.8 else 'moderado' if abs(cohens_d) > 0.5 else 'pequeño'}."
        )
        
        recommendations = []
        if not assumptions.get("group1_normal", True) or not assumptions.get("group2_normal", True):
            recommendations.append("Considerar pruebas no paramétricas (Mann-Whitney U)")
        if abs(cohens_d) < 0.2:
            recommendations.append("El tamaño del efecto es muy pequeño, considerar relevancia práctica")
        
        return StatisticalResult(
            test_name=test_name,
            statistic=float(t_stat),
            p_value=float(p_val),
            df=df,
            effect_size=float(cohens_d),
            ci_95=(float(ci[0]), float(ci[1])),
            interpretation=interpretation,
            assumptions=assumptions,
            recommendations=recommendations
        )
    
    def anova(self, groups: List[np.ndarray], test_type: str = "one_way") -> StatisticalResult:
        """Análisis de varianza (ANOVA)"""
        # [Mantener el resto del método...]
        # Verificar supuestos
        assumptions = {
            "normality": all(len(g) >= 3 for g in groups),
            "equal_variance": True,
            "independence": True
        }
        
        # Test de normalidad para cada grupo
        normality_results = []
        for i, group in enumerate(groups):
            if 3 <= len(group) <= 5000:
                _, p = stats.shapiro(group)
                normality_results.append(p > 0.05)
            else:
                normality_results.append(True)  # Asumir normalidad para n grande
        
        assumptions["all_groups_normal"] = all(normality_results)
        
        # Test de homogeneidad de varianzas (Levene)
        if all(len(g) > 1 for g in groups):
            _, p_levene = stats.levene(*groups)
            assumptions["equal_variance"] = p_levene > 0.05
        
        # Realizar ANOVA
        if test_type == "one_way":
            f_stat, p_val = stats.f_oneway(*groups)
            test_name = "One-way ANOVA"
            df_between = len(groups) - 1
            df_within = sum(len(g) for g in groups) - len(groups)
            df = (df_between, df_within)
        else:
            # Para ANOVA de dos vías necesitaríamos datos estructurados
            raise NotImplementedError("ANOVA de dos vías no implementada aún")
        
        # Calcular tamaño del efecto (eta-squared)
        ss_total = sum((x - np.mean(np.concatenate(groups)))**2 for x in np.concatenate(groups))
        ss_between = sum(len(g) * (np.mean(g) - np.mean(np.concatenate(groups)))**2 for g in groups)
        eta_squared = ss_between / ss_total if ss_total != 0 else 0
        
        # Interpretación
        if p_val < 0.001:
            significance = "altamente significativas"
        elif p_val < 0.01:
            significance = "muy significativas"
        elif p_val < 0.05:
            significance = "significativas"
        else:
            significance = "no significativas"
        
        interpretation = (
            f"Existen diferencias {significance} entre los grupos (p = {p_val:.4f}). "
            f"El tamaño del efecto (η² = {eta_squared:.3f}) indica que {'una gran' if eta_squared > 0.14 else 'una moderada' if eta_squared > 0.06 else 'una pequeña'} "
            f"proporción de la varianza es explicada por las diferencias entre grupos."
        )
        
        recommendations = []
        if not assumptions["all_groups_normal"]:
            recommendations.append("Considerar Kruskal-Wallis (ANOVA no paramétrica)")
        if not assumptions["equal_variance"]:
            recommendations.append("Considerar corrección de Welch para ANOVA")
        if p_val < 0.05:
            recommendations.append("Realizar pruebas post-hoc para comparaciones múltiples")
        
        return StatisticalResult(
            test_name=test_name,
            statistic=float(f_stat),
            p_value=float(p_val),
            df=df,
            effect_size=float(eta_squared),
            interpretation=interpretation,
            assumptions=assumptions,
            recommendations=recommendations
        )
    
    def correlation_analysis(self, x: np.ndarray, y: np.ndarray, 
                            method: str = "pearson") -> StatisticalResult:
        """Análisis de correlación"""
        # [Mantener el resto del método...]
        # Verificar supuestos
        assumptions = {
            "paired_data": len(x) == len(y),
            "normality": True
        }
        
        if len(x) != len(y):
            raise ValueError("Los arrays x e y deben tener la misma longitud")
        
        # Test de normalidad
        if 3 <= len(x) <= 5000:
            _, p_x = stats.shapiro(x)
            _, p_y = stats.shapiro(y)
            assumptions["x_normal"] = p_x > 0.05
            assumptions["y_normal"] = p_y > 0.05
            assumptions["normality"] = p_x > 0.05 and p_y > 0.05
        
        # Realizar correlación
        if method == "pearson":
            corr, p_val = stats.pearsonr(x, y)
            test_name = "Pearson correlation"
        elif method == "spearman":
            corr, p_val = stats.spearmanr(x, y)
            test_name = "Spearman correlation"
        elif method == "kendall":
            corr, p_val = stats.kendalltau(x, y)
            test_name = "Kendall's tau"
        else:
            raise ValueError(f"Método {method} no válido. Usar 'pearson', 'spearman' o 'kendall'")
        
        # Intervalo de confianza 95% para correlación
        if method == "pearson" and len(x) > 3:
            z = np.arctanh(corr)
            se = 1 / np.sqrt(len(x) - 3)
            ci_z = (z - 1.96*se, z + 1.96*se)
            ci = (np.tanh(ci_z[0]), np.tanh(ci_z[1]))
        else:
            ci = None
        
        # Interpretación de la fuerza
        abs_corr = abs(corr)
        if abs_corr >= 0.9:
            strength = "muy fuerte"
        elif abs_corr >= 0.7:
            strength = "fuerte"
        elif abs_corr >= 0.5:
            strength = "moderada"
        elif abs_corr >= 0.3:
            strength = "débil"
        else:
            strength = "muy débil o nula"
        
        # Interpretación
        if p_val < 0.001:
            significance = "altamente significativa"
        elif p_val < 0.01:
            significance = "muy significativa"
        elif p_val < 0.05:
            significance = "significativa"
        else:
            significance = "no significativa"
        
        direction = "positiva" if corr > 0 else "negativa"
        interpretation = (
            f"La correlación {direction} es de {strength} (r = {corr:.3f}) y {significance} (p = {p_val:.4f})."
        )
        
        recommendations = []
        if method == "pearson" and not assumptions.get("normality", True):
            recommendations.append("Considerar correlación de Spearman (no paramétrica)")
        if abs_corr < 0.3 and p_val < 0.05:
            recommendations.append("La correlación es significativa pero débil, considerar relevancia práctica")
        
        return StatisticalResult(
            test_name=test_name,
            statistic=float(corr),
            p_value=float(p_val),
            ci_95=ci,
            effect_size=float(corr),
            interpretation=interpretation,
            assumptions=assumptions,
            recommendations=recommendations
        )
    
    def regression_analysis(self, X: np.ndarray, y: np.ndarray, 
                           include_intercept: bool = True) -> Dict:
        """Análisis de regresión lineal"""
        if not HAS_STATSMODELS:
            return {"error": "statsmodels no está instalado"}
        
        # Añadir intercepto si se solicita
        if include_intercept:
            X = sm.add_constant(X)
        
        # Ajustar modelo
        model = sm.OLS(y, X).fit()
        
        # Extraer resultados
        results = {
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "f_statistic": float(model.fvalue),
            "f_p_value": float(model.f_pvalue),
            "aic": float(model.aic),
            "bic": float(model.bic),
            "residuals_info": {
                "mean": float(model.resid.mean()),
                "std": float(model.resid.std()),
                "min": float(model.resid.min()),
                "max": float(model.resid.max())
            },
            "coefficients": []
        }
        
        # Coeficientes
        for i, coef_name in enumerate(model.params.index):
            results["coefficients"].append({
                "variable": str(coef_name),
                "coefficient": float(model.params[i]),
                "std_error": float(model.bse[i]),
                "t_stat": float(model.tvalues[i]),
                "p_value": float(model.pvalues[i]),
                "ci_95_lower": float(model.conf_int()[0][i]),
                "ci_95_upper": float(model.conf_int()[1][i])
            })
        
        # Diagnósticos
        if len(y) > 10:
            # Test de normalidad de residuos
            _, jb_p = stats.jarque_bera(model.resid)
            _, shapiro_p = stats.shapiro(model.resid) if len(model.resid) <= 5000 else (None, None)
            
            # Durbin-Watson para autocorrelación
            dw = sm.stats.stattools.durbin_watson(model.resid)
            
            results["diagnostics"] = {
                "jarque_bera_p": float(jb_p),
                "shapiro_p": float(shapiro_p) if shapiro_p else None,
                "durbin_watson": float(dw),
                "residuals_normal": jb_p > 0.05,
                "no_autocorrelation": 1.5 < dw < 2.5
            }
        
        return results
    
    def save_results(self, filename: str = "statistical_results.json"):
        """Guardar resultados en archivo"""
        with open(filename, "w") as f:
            json.dump([r.to_dict() for r in self.results_history], f, indent=2)
    
    def load_results(self, filename: str = "statistical_results.json"):
        """Cargar resultados desde archivo"""
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                # Reconstruir objetos StatisticalResult
                self.results_history = [
                    StatisticalResult(**item) for item in data
                ]
        except FileNotFoundError:
            print(f"Archivo {filename} no encontrado")


# Instancia global
statistical_engine = StatisticalPhDEngine()
