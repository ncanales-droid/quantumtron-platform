"""MLflow Model Registry Manager for QuantumTron."""

import mlflow
from mlflow.exceptions import MlflowException
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class ModelRegistryManager:
    """Manager for MLflow Model Registry operations."""
    
    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db"):
        """Initialize Model Registry Manager.
        
        Args:
            tracking_uri: MLflow tracking URI
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.MlflowClient()
        
    def register_model(
        self,
        run_id: str,
        model_name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Register a model from a run to the Model Registry.
        
        Args:
            run_id: ID of the MLflow run
            model_name: Name for the registered model
            description: Optional description
            tags: Optional tags for the model
            
        Returns:
            Version number (e.g., "1")
        """
        try:
            # Create model if it doesn't exist
            try:
                self.client.get_registered_model(model_name)
            except MlflowException:
                self.client.create_registered_model(
                    name=model_name,
                    description=description,
                    tags=tags
                )
                logger.info(f"Created new registered model: {model_name}")
            
            # Register the model version
            model_uri = f"runs:/{run_id}/model"
            model_version = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id
            )
            
            logger.info(f"Registered model: {model_name} version {model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = False
    ) -> None:
        """Transition a model version to a new stage.
        
        Args:
            model_name: Name of the registered model
            version: Version number (string)
            stage: Target stage ("Staging", "Production", "Archived")
            archive_existing_versions: Whether to archive existing versions in the target stage
        """
        valid_stages = ["Staging", "Production", "Archived", "None"]
        if stage not in valid_stages:
            raise ValueError(f"Stage must be one of {valid_stages}")
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing_versions
            )
            logger.info(f"Transitioned {model_name} version {version} to {stage}")
        except Exception as e:
            logger.error(f"Error transitioning model stage: {e}")
            raise
    
    def get_model_version(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Any:
        """Get a model version by version number or stage.
        
        Args:
            model_name: Name of the registered model
            version: Specific version number
            stage: Stage to get latest version from
            
        Returns:
            Model version object
        """
        if version and stage:
            raise ValueError("Specify either version or stage, not both")
        
        try:
            if version:
                return self.client.get_model_version(model_name, version)
            elif stage:
                return self.client.get_latest_versions(model_name, stages=[stage])[0]
            else:
                # Get latest version
                versions = self.client.get_latest_versions(model_name, stages=[])
                return versions[0] if versions else None
                
        except Exception as e:
            logger.error(f"Error getting model version: {e}")
            raise
    
    def load_model_for_inference(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Any:
        """Load a registered model for inference.
        
        Args:
            model_name: Name of the registered model
            version: Specific version number
            stage: Stage to load from
            
        Returns:
            Loaded model ready for inference
        """
        model_version = self.get_model_version(model_name, version, stage)
        if not model_version:
            raise ValueError(f"Model {model_name} not found")
        
        model_uri = f"models:/{model_name}/{model_version.version}"
        return mlflow.pyfunc.load_model(model_uri)
    
    def get_all_registered_models(self) -> List[Dict[str, Any]]:
        """Get all registered models with their versions."""
        try:
            models = self.client.search_registered_models()
            result = []
            
            for model in models:
                model_info = {
                    'name': model.name,
                    'description': model.description,
                    'tags': model.tags,
                    'versions': []
                }
                
                # Get all versions
                versions = self.client.search_model_versions(f"name='{model.name}'")
                for version in versions:
                    version_info = {
                        'version': version.version,
                        'current_stage': version.current_stage,
                        'run_id': version.run_id,
                        'status': version.status,
                        'created_at': version.creation_timestamp,
                        'last_updated': version.last_updated_timestamp
                    }
                    model_info['versions'].append(version_info)
                
                result.append(model_info)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting registered models: {e}")
            return []
    
    def compare_model_versions(
        self,
        model_name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Compare two versions of the same model.
        
        Args:
            model_name: Name of the registered model
            version1: First version to compare
            version2: Second version to compare
            
        Returns:
            Comparison results
        """
        try:
            v1 = self.client.get_model_version(model_name, version1)
            v2 = self.client.get_model_version(model_name, version2)
            
            # Get run information for both versions
            run1 = mlflow.get_run(v1.run_id)
            run2 = mlflow.get_run(v2.run_id)
            
            comparison = {
                'model_name': model_name,
                'versions': {
                    version1: {
                        'run_id': v1.run_id,
                        'stage': v1.current_stage,
                        'metrics': run1.data.metrics,
                        'params': run1.data.params,
                        'created_at': v1.creation_timestamp
                    },
                    version2: {
                        'run_id': v2.run_id,
                        'stage': v2.current_stage,
                        'metrics': run2.data.metrics,
                        'params': run2.data.params,
                        'created_at': v2.creation_timestamp
                    }
                },
                'differences': {}
            }
            
            # Compare metrics
            metrics_diff = {}
            for metric in set(run1.data.metrics.keys()) | set(run2.data.metrics.keys()):
                val1 = run1.data.metrics.get(metric)
                val2 = run2.data.metrics.get(metric)
                if val1 != val2:
                    metrics_diff[metric] = {
                        'version1': val1,
                        'version2': val2,
                        'difference': val2 - val1 if val1 and val2 else None
                    }
            
            if metrics_diff:
                comparison['differences']['metrics'] = metrics_diff
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing model versions: {e}")
            raise
    
    def set_model_description(
        self,
        model_name: str,
        version: str,
        description: str
    ) -> None:
        """Set description for a model version.
        
        Args:
            model_name: Name of the registered model
            version: Version number
            description: Description text
        """
        try:
            self.client.update_model_version(
                name=model_name,
                version=version,
                description=description
            )
            logger.info(f"Updated description for {model_name} version {version}")
        except Exception as e:
            logger.error(f"Error setting model description: {e}")
            raise
    
    def delete_model_version(
        self,
        model_name: str,
        version: str
    ) -> None:
        """Delete a model version.
        
        Args:
            model_name: Name of the registered model
            version: Version number to delete
        """
        try:
            self.client.delete_model_version(
                name=model_name,
                version=version
            )
            logger.info(f"Deleted {model_name} version {version}")
        except Exception as e:
            logger.error(f"Error deleting model version: {e}")
            raise
    
    def search_models(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """Search for registered models.
        
        Args:
            filter_string: Filter string (e.g., "name LIKE 'xgboost%'")
            max_results: Maximum number of results
            
        Returns:
            List of matching models
        """
        try:
            models = self.client.search_registered_models(
                filter_string=filter_string,
                max_results=max_results
            )
            
            return [
                {
                    'name': model.name,
                    'description': model.description,
                    'latest_versions': [
                        {
                            'version': v.version,
                            'stage': v.current_stage,
                            'run_id': v.run_id
                        }
                        for v in model.latest_versions
                    ]
                }
                for model in models
            ]
            
        except Exception as e:
            logger.error(f"Error searching models: {e}")
            return []


# Singleton instance
model_registry_manager = ModelRegistryManager()
