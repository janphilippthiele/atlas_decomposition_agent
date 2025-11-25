"""
Configuration loader module for database dependency analysis tool.

This module provides functionality to load, validate, and access
configuration from YAML files with support for overrides and runtime updates.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from datetime import datetime
import copy


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass


class ConfigValidationError(ConfigError):
    """Exception raised when configuration validation fails."""
    pass


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ConfigLoader:
    """
    Loads and manages configuration from YAML files.
    
    This class provides:
    - Loading configuration from YAML files
    - Validation of configuration structure
    - Environment-specific overrides
    - Easy access to configuration values
    - Runtime configuration updates
    """
    
    REQUIRED_SECTIONS = {
        'parsing_rules.yaml': ['languages', 'object_hierarchy', 'global'],
        'file_patterns.yaml': ['file_mappings', 'exclusions', 'processing_constraints']
    }
    
    def __init__(self, config_dir: Union[str, Path], 
                 environment: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
            environment: Environment name for overrides (dev, test, prod)
        """
        self.config_dir = Path(r"")
        self.environment = environment or os.getenv('APP_ENV', 'default')
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configuration storage
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._override_configs: Dict[str, Dict[str, Any]] = {}
        self._merged_config: Dict[str, Any] = {}
        
        # Validation cache
        self._validation_cache: Dict[str, ValidationResult] = {}
        
        # Load configurations
        self.load_all_configs()
    
    def load_all_configs(self) -> None:
        """Load all configuration files."""
        try:
            # Load main configuration files
            self._configs['parsing_rules'] = self._load_yaml_file('parsing_rules.yaml')
            self._configs['file_patterns'] = self._load_yaml_file('file_patterns.yaml')
            
            # Load environment-specific overrides if they exist
            if self.environment and self.environment != 'default':
                self._load_environment_overrides()
            
            # Merge configurations
            self._merge_configurations()
            
            # Validate loaded configurations
            validation_result = self.validate_all()
            if not validation_result.is_valid:
                raise ConfigValidationError(
                    f"Configuration validation failed: {'; '.join(validation_result.errors)}"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to load configurations: {str(e)}")
            raise
    
    def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """Load a single YAML file."""
        file_path = self.config_dir / filename
        self.logger.debug(f"Trying to load config file at: {file_path.resolve()}")  # Add this line

        if not file_path.exists():
            raise ConfigError(f"Configuration file not found: {file_path.resolve()}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            if config is None:
                config = {}
                
            self.logger.info(f"Loaded configuration from {file_path}")
            return config
            
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse YAML file {filename}: {str(e)}")
        except Exception as e:
            raise ConfigError(f"Failed to read file {filename}: {str(e)}")
    
    def _load_environment_overrides(self) -> None:
        """Load environment-specific override configurations."""
        override_dir = self.config_dir / 'environments' / self.environment
        
        if not override_dir.exists():
            self.logger.info(f"No environment overrides found for '{self.environment}'")
            return
        
        for yaml_file in override_dir.glob('*.yaml'):
            config_name = yaml_file.stem
            try:
                override_config = self._load_yaml_file(
                    f'environments/{self.environment}/{yaml_file.name}'
                )
                self._override_configs[config_name] = override_config
                self.logger.info(f"Loaded override configuration: {yaml_file.name}")
            except Exception as e:
                self.logger.warning(f"Failed to load override {yaml_file}: {str(e)}")
    
    def _merge_configurations(self) -> None:
        """Merge base configurations with overrides."""
        self._merged_config = copy.deepcopy(self._configs)
        
        # Apply overrides
        for config_name, overrides in self._override_configs.items():
            if config_name in self._merged_config:
                self._merged_config[config_name] = self._deep_merge(
                    self._merged_config[config_name],
                    overrides
                )
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            config.get('parsing_rules.languages.SQL.comments.single_line')
        """
        keys = key_path.split('.')
        value = self._merged_config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_language_config(self, language: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific language."""
        return self.get(f'parsing_rules.languages.{language}')
    
    def get_file_extensions(self, language: str) -> List[str]:
        """Get file extensions for a specific language."""
        lang_config = self.get_language_config(language)
        if lang_config:
            return lang_config.get('file_extensions', [])
        return []
    
    def get_exclusion_patterns(self) -> Dict[str, Any]:
        """Get file and directory exclusion patterns."""
        return self.get('file_patterns.exclusions', {})
    
    def get_object_patterns(self, language: str, object_type: str) -> Dict[str, Any]:
        """Get object patterns for a specific language and object type."""
        return self.get(f'parsing_rules.languages.{language}.object_patterns.{object_type}', {})
    
    def get_duplicate_thresholds(self) -> Dict[str, float]:
        """Get duplicate detection thresholds."""
        return self.get('file_patterns.duplicate_detection.thresholds', {})
    
    def get_processing_limits(self) -> Dict[str, Any]:
        """Get file processing limits."""
        return self.get('file_patterns.processing_constraints.limits', {})
    
    def validate_all(self) -> ValidationResult:
        """Validate all loaded configurations."""
        errors = []
        warnings = []
        
        # Validate parsing rules
        pr_result = self._validate_parsing_rules()
        errors.extend(pr_result.errors)
        warnings.extend(pr_result.warnings)
        
        # Validate file patterns
        fp_result = self._validate_file_patterns()
        errors.extend(fp_result.errors)
        warnings.extend(fp_result.warnings)
        
        # Cross-validate configurations
        cross_result = self._cross_validate()
        errors.extend(cross_result.errors)
        warnings.extend(cross_result.warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_parsing_rules(self) -> ValidationResult:
        """Validate parsing rules configuration."""
        errors = []
        warnings = []
        config = self._merged_config.get('parsing_rules', {})
        
        # Check required sections
        for section in self.REQUIRED_SECTIONS['parsing_rules.yaml']:
            if section not in config:
                errors.append(f"Missing required section in parsing_rules: {section}")
        
        # Validate languages
        languages = config.get('languages', {})
        if not languages:
            errors.append("No languages defined in parsing_rules")
        
        for lang_name, lang_config in languages.items():
            # Check required language fields
            if 'file_extensions' not in lang_config:
                errors.append(f"Language '{lang_name}' missing file_extensions")
            
            # Validate regex patterns
            patterns = lang_config.get('object_patterns', {})
            for obj_type, obj_patterns in patterns.items():
                for pattern_name, pattern_value in obj_patterns.items():
                    if isinstance(pattern_value, str):
                        try:
                            re.compile(pattern_value)
                        except re.error as e:
                            errors.append(
                                f"Invalid regex in {lang_name}.{obj_type}.{pattern_name}: {str(e)}"
                            )
                    elif isinstance(pattern_value, list):
                        for idx, pattern in enumerate(pattern_value):
                            try:
                                re.compile(pattern)
                            except re.error as e:
                                errors.append(
                                    f"Invalid regex in {lang_name}.{obj_type}.{pattern_name}[{idx}]: {str(e)}"
                                )
        
        # Validate confidence thresholds
        global_config = config.get('global', {})
        quality = global_config.get('quality', {})
        
        # Check file size limits
        max_size = global_config.get('max_file_size_mb', 0)
        if max_size <= 0:
            warnings.append("No maximum file size limit set")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_file_patterns(self) -> ValidationResult:
        """Validate file patterns configuration."""
        errors = []
        warnings = []
        config = self._merged_config.get('file_patterns', {})
        
        # Check required sections
        for section in self.REQUIRED_SECTIONS['file_patterns.yaml']:
            if section not in config:
                errors.append(f"Missing required section in file_patterns: {section}")
        
        # Validate file mappings
        mappings = config.get('file_mappings', {})
        if not mappings:
            errors.append("No file mappings defined")
        
        # Check for extension conflicts
        extension_map = {}
        for lang, lang_config in mappings.items():
            extensions = lang_config.get('extensions', [])
            for ext in extensions:
                if ext in extension_map:
                    warnings.append(
                        f"Extension '{ext}' mapped to multiple languages: "
                        f"{extension_map[ext]} and {lang}"
                    )
                extension_map[ext] = lang
        
        # Validate exclusion patterns
        exclusions = config.get('exclusions', {})
        for pattern in exclusions.get('directories', {}).get('patterns', []):
            # Basic validation of glob patterns
            if not pattern:
                errors.append("Empty directory exclusion pattern")
        
        # Validate duplicate detection settings
        dup_config = config.get('duplicate_detection', {})
        if dup_config.get('enabled', False):
            thresholds = dup_config.get('thresholds', {})
            for name, value in thresholds.items():
                if not 0 <= value <= 1:
                    errors.append(f"Invalid threshold value for {name}: {value}")
        
        # Validate processing constraints
        constraints = config.get('processing_constraints', {})
        file_size = constraints.get('file_size', {})
        if file_size.get('max_size_mb', 0) <= 0:
            warnings.append("No maximum file size constraint set")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _cross_validate(self) -> ValidationResult:
        """Cross-validate between different configuration files."""
        errors = []
        warnings = []
        
        # Get languages from parsing rules
        parsing_languages = set(
            self.get('parsing_rules.languages', {}).keys()
        )
        
        # Get languages from file patterns
        file_languages = set(
            self.get('file_patterns.file_mappings', {}).keys()
        )
        
        # Check for mismatches
        in_parsing_only = parsing_languages - file_languages
        in_files_only = file_languages - parsing_languages
        
        for lang in in_parsing_only:
            warnings.append(f"Language '{lang}' defined in parsing_rules but not in file_patterns")
        
        for lang in in_files_only:
            warnings.append(f"Language '{lang}' defined in file_patterns but not in parsing_rules")
        
        return ValidationResult(
            is_valid=True,  # Cross-validation issues are warnings, not errors
            errors=errors,
            warnings=warnings
        )
    
    def update_config(self, key_path: str, value: Any) -> None:
        """
        Update configuration value at runtime.
        
        Args:
            key_path: Dot-separated path to configuration value
            value: New value to set
        """
        keys = key_path.split('.')
        config = self._merged_config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
        
        # Clear validation cache
        self._validation_cache.clear()
        
        self.logger.info(f"Updated configuration: {key_path} = {value}")
    
    def reload(self) -> None:
        """Reload all configuration files."""
        self._configs.clear()
        self._override_configs.clear()
        self._merged_config.clear()
        self._validation_cache.clear()
        
        self.load_all_configs()
        self.logger.info("Configuration reloaded")
    
    def export_config(self, output_path: Path, 
                     include_defaults: bool = True) -> None:
        """
        Export current configuration to file.
        
        Args:
            output_path: Path to export configuration
            include_defaults: Whether to include default values
        """
        config_to_export = self._merged_config if include_defaults else self._override_configs
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_to_export, f, default_flow_style=False, sort_keys=True)
        
        self.logger.info(f"Exported configuration to {output_path}")
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get information about loaded configuration."""
        return {
            'environment': self.environment,
            'config_dir': str(self.config_dir),
            'loaded_files': list(self._configs.keys()),
            'has_overrides': len(self._override_configs) > 0,
            'override_files': list(self._override_configs.keys()),
            'last_reload': datetime.now().isoformat()
        }
    
    def __repr__(self) -> str:
        """String representation of configuration loader."""
        return (
            f"ConfigLoader(config_dir='{self.config_dir}', "
            f"environment='{self.environment}', "
            f"loaded_configs={list(self._configs.keys())})"
        )