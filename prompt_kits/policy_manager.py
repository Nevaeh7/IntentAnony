"""
Policy Manager - File-driven version

Features:
 Automatically load JSON configurations from policy_prompts/ folder
 Support JSONSchema validation
 Support multi-language policy configurations
 Hot reload (optional)
 Provide convenient intent and attribute query interfaces
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Union
from pathlib import Path
import json
from loguru import logger

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    logger.warning("jsonschema not installed. Install with: pip install jsonschema")


# JSONSchema for policy configuration
POLICY_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "version": {"type": "string"},
        "locale": {"type": "string"},
        "module": {"type": "string"},
        "principle": {"type": "string"},
        "levels": {"type": "object"},
        "actions": {"type": "array"},
        "attributes": {"type": "array"},
        "attributes_desc": {"type": "object"},
        "intents": {"type": "object"},
        "intent_attribute_allow_matrix": {"type": "object"},
        "intent_combination": {"type": "object"},
        "ban_rules": {"type": "object"},
        "combo_rule": {"type": "object"},
        "action_policy": {"type": "object"},
        "presets": {"type": "object"},
        "execution_flow": {"type": "array"},
        "example": {"type": "object"}
    },
    "required": ["version", "attributes"]
}


@dataclass
class PolicyConfig:
    """Policy configuration"""
    version: str
    locale: str
    module: str
    principle: str = ""
    levels: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    attributes_desc: Dict[str, str] = field(default_factory=dict)
    intents: Dict[str, Any] = field(default_factory=dict)
    intent_matrix: Dict[str, Dict[str, str]] = field(default_factory=dict)
    intent_combination: Dict[str, Any] = field(default_factory=dict)
    ban_rules: Dict[str, Any] = field(default_factory=dict)
    combo_rule: Dict[str, Any] = field(default_factory=dict)
    action_policy: Dict[str, Any] = field(default_factory=dict)
    presets: Dict[str, Dict[str, str]] = field(default_factory=dict)
    execution_flow: List[str] = field(default_factory=list)
    output_contract: Dict[str, Any] = field(default_factory=dict)
    example: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[Path] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_max_level(self, intent: str, attribute: str) -> Optional[str]:
        """
        Get the maximum granularity limit for an attribute under a specified intent
        
        Args:
            intent: Intent identifier (I1-I5)
            attribute: Attribute identifier (AGE, EDU, SEX, OCC, MAR, LOC, POB, INC)
            
        Returns:
            Maximum granularity level (L0-L3 or BAN), returns None if not exists
        """
        intent_config = self.intent_matrix.get(intent, {})
        return intent_config.get(attribute)
    
    def get_max_levels_for_intents(self, intents: Dict[str, float]) -> Dict[str, str]:
        """
        Calculate maximum granularity limit for each attribute based on intent vector (with weights)
        Uses most conservative strategy (takes lowest allowed granularity)
        
        Args:
            intents: Intent vector, format like {"I1": 0.6, "I2": 0.4}
            
        Returns:
            Dictionary of maximum granularity limits for each attribute
        """
        if not intents:
            return {}
        
        # Get all involved intents
        intent_list = list(intents.keys())
        
        # Initialize result dictionary
        max_levels = {}
        attribute_list = self.attributes
        
        # For each attribute, take the most conservative granularity from all intents
        for attr in attribute_list:
            levels = []
            for intent in intent_list:
                level = self.get_max_level(intent, attr)
                if level:
                    levels.append(level)
            
            if levels:
                # Sort by granularity level, take the most conservative (highest rank)
                # BAN > L3 > L2 > L1 > L0
                level_ranks = {
                    "BAN": 4,
                    "L3": 3,
                    "L2": 2,
                    "L1": 1,
                    "L0": 0
                }
                
                # Get the most conservative granularity for this attribute across all intents
                max_levels[attr] = max(levels, key=lambda x: level_ranks.get(x, -1))
            else:
                max_levels[attr] = "BAN"  # Default most conservative
        
        return max_levels
    
    def get_preset(self, preset_name: str) -> Optional[Dict[str, str]]:
        """
        Get preset configuration
        
        Args:
            preset_name: Preset name (e.g., "social_default", "work_share", "strict")
            
        Returns:
            Preset granularity configuration dictionary, returns None if not exists
        """
        return self.presets.get(preset_name)
    
    def get_intent_info(self, intent: str) -> Optional[Dict[str, Any]]:
        """
        Get intent information
        
        Args:
            intent: Intent identifier (I1-I5)
            
        Returns:
            Intent information dictionary, returns None if not exists
        """
        return self.intents.get(intent)
    
    def get_attribute_desc(self, attribute: str) -> Optional[str]:
        """
        Get attribute description
        
        Args:
            attribute: Attribute identifier (AGE, EDU, SEX, OCC, MAR, LOC, POB, INC)
            
        Returns:
            Attribute description, returns None if not exists
        """
        return self.attributes_desc.get(attribute)


class PolicyManager:
    """Policy Manager - Load JSON configurations from file system"""
    
    def __init__(self, policies_dir: Optional[Path] = None, auto_reload: bool = False):
        self.policies_dir = policies_dir or Path(__file__).parent / "policy_prompts"
        self.auto_reload = auto_reload
        self._policies: Dict[str, Dict[str, PolicyConfig]] = {}  # version -> language -> PolicyConfig
        self._file_timestamps: Dict[str, float] = {}
        
        # Ensure directory exists
        self.policies_dir.mkdir(exist_ok=True)
        
        # Load all policy configurations
        self.load_all()
    
    def load_all(self):
        """Load all JSON configuration files"""
        policy_files = list(self.policies_dir.glob("*.json"))
        
        for file_path in policy_files:
            try:
                self.load_from_file(file_path)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(self._policies)} policy configurations (total {len(policy_files)} files)")
    
    def load_from_file(self, file_path: Path) -> PolicyConfig:
        """Load policy configuration from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate configuration
        if JSONSCHEMA_AVAILABLE:
            try:
                jsonschema.validate(instance=data, schema=POLICY_CONFIG_SCHEMA)
            except jsonschema.exceptions.ValidationError as e:
                logger.error(f"Configuration validation failed {file_path}: {e}")
        
        # Extract language identifier from locale (e.g., "zh-CN" -> "zh")
        locale = data.get('locale', 'zh')
        language = locale.split('-')[0] if '-' in locale else locale
        version = data.get('version', '1.0')
        
        # Create configuration object
        config = PolicyConfig(
            version=version,
            locale=data.get('locale', language),
            module=data.get('module', 'Privacy Policy'),
            principle=data.get('principle', ''),
            levels=data.get('levels', {}),
            actions=data.get('actions', []),
            attributes=data.get('attributes', []),
            attributes_desc=data.get('attributes_desc', {}),
            intents=data.get('intents', {}),
            intent_matrix=data.get('intent_matrix', {}),
            intent_combination=data.get('intent_combination', {}),
            ban_rules=data.get('ban_rules', {}),
            combo_rule=data.get('combo_rule', {}),
            action_policy=data.get('action_policy', {}),
            presets=data.get('presets', {}),
            execution_flow=data.get('execution_flow', []),
            output_contract=data.get('output_contract', {}),
            example=data.get('example', {}),
            file_path=file_path,
            raw_data=data
        )
        
        # Register policy (by version and language)
        self.register(version, language, config)
        
        # Record file timestamp
        self._file_timestamps[str(file_path)] = file_path.stat().st_mtime
        
        return config
    
    def register(self, version: str, language: str, config: PolicyConfig):
        """Register policy configuration"""
        if version not in self._policies:
            self._policies[version] = {}
        self._policies[version][language] = config
        # logger.debug(f"Registered policy configuration: version={version}, language={language}")
    
    def check_reload(self, version: str, language: str = "zh"):
        """Check and reload file (if auto-reload is enabled)"""
        if not self.auto_reload:
            return
        
        version_policies = self._policies.get(version)
        if not version_policies:
            return
        
        config = version_policies.get(language)
        if not config or not config.file_path:
            return
        
        file_path = config.file_path
        current_mtime = file_path.stat().st_mtime
        cached_mtime = self._file_timestamps.get(str(file_path), 0)
        
        if current_mtime > cached_mtime:
            logger.info(f"File change detected, reloading: {file_path}")
            self.load_from_file(file_path)
    
    def get(self, version: str = None, language: str = "zh") -> PolicyConfig:
        """
        Get policy configuration
        
        Args:
            version: Version number (e.g., "3.0"), if None returns latest version
            language: Language identifier (e.g., "zh", "en")
            
        Returns:
            Policy configuration object
        """
        # If version not specified, use latest version
        if version is None:
            if not self._policies:
                raise ValueError("No available policy configurations")
            # Use smarter version comparison (supports semantic versioning)
            try:
                from packaging import version as pkg_version
                version = max(self._policies.keys(), key=lambda v: pkg_version.parse(v))
            except ImportError:
                # If packaging library not available, use simple string comparison
                version = max(self._policies.keys())
            logger.debug(f"Version not specified, using latest version: {version}")
        
        self.check_reload(version, language)
        
        if version not in self._policies:
            available_versions = list(self._policies.keys())
            raise ValueError(f"Version not found: {version}, available versions: {available_versions}")
        
        if language not in self._policies[version]:
            available_languages = list(self._policies[version].keys())
            raise ValueError(f"Language not found: {language} (version: {version}), available languages: {available_languages}")
        
        return self._policies[version][language]
    
    def get_max_level(self, intent: str, attribute: str, version: str = None, language: str = "zh") -> Optional[str]:
        """Get maximum granularity limit for an attribute under specified intent"""
        config = self.get(version, language)
        return config.get_max_level(intent, attribute)
    
    def get_max_levels_for_intents(self, intents: Dict[str, float], version: str = None, language: str = "zh") -> Dict[str, str]:
        """Calculate maximum granularity limit for each attribute based on intent vector"""
        config = self.get(version, language)
        return config.get_max_levels_for_intents(intents)
    
    def get_preset(self, preset_name: str, version: str = None, language: str = "zh") -> Optional[Dict[str, str]]:
        """Get preset configuration"""
        config = self.get(version, language)
        return config.get_preset(preset_name)
    
    def get_intent_info(self, intent: str, version: str = None, language: str = "zh") -> Optional[Dict[str, Any]]:
        """Get intent information"""
        config = self.get(version, language)
        return config.get_intent_info(intent)
    
    def get_attribute_desc(self, attribute: str, version: str = None, language: str = "zh") -> Optional[str]:
        """Get attribute description"""
        config = self.get(version, language)
        return config.get_attribute_desc(attribute)
    
    def list_versions(self) -> List[str]:
        """List all supported versions"""
        return list(self._policies.keys())
    
    def list_languages(self, version: str = None) -> List[str]:
        """
        List all supported languages
        
        Args:
            version: Version number, if None returns union of all versions
        """
        if version:
            if version not in self._policies:
                return []
            return list(self._policies[version].keys())
        else:
            # Return union of all versions
            all_languages = set()
            for version_policies in self._policies.values():
                all_languages.update(version_policies.keys())
            return sorted(list(all_languages))
    
    def get_latest_version(self) -> Optional[str]:
        """Get latest version number"""
        if not self._policies:
            return None
        # Use smarter version comparison (supports semantic versioning)
        try:
            from packaging import version as pkg_version
            return max(self._policies.keys(), key=lambda v: pkg_version.parse(v))
        except ImportError:
            # If packaging library not available, use simple string comparison
            return max(self._policies.keys())
    
    def reload(self, version: str = None, language: str = None):
        """
        Manually reload policy configuration
        
        Args:
            version: Version number
                - If both version and language provided: reload specific version and language policy
                - If only version provided: reload all languages for that version
                - If neither provided: reload all policies
            language: Language identifier
        """
        if version and language:
            # Reload specific version and language
            version_policies = self._policies.get(version)
            if version_policies:
                config = version_policies.get(language)
                if config and config.file_path:
                    logger.info(f"Reloading policy: version={version}, language={language}")
                    self.load_from_file(config.file_path)
                else:
                    logger.warning(f"Policy not found: version={version}, language={language}")
            else:
                logger.warning(f"Version not found: {version}")
        elif version:
            # Reload all languages for specific version
            version_policies = self._policies.get(version)
            if version_policies:
                logger.info(f"Reloading all languages for version {version}")
                for lang, config in version_policies.items():
                    if config and config.file_path:
                        self.load_from_file(config.file_path)
            else:
                logger.warning(f"Version not found: {version}")
        else:
            # Reload all
            logger.info("Reloading all policy configurations")
            self._policies.clear()
            self._file_timestamps.clear()
            self.load_all()
    
    def get_as_json_string(self, version: str = None, language: str = "zh") -> str:
        """Get policy configuration as JSON string (for passing to prompts)"""
        config = self.get(version, language)
        return json.dumps(config.raw_data, ensure_ascii=False, indent=2)


# Global instance
_global_policy_manager = None

def get_policy_manager(force_reload: bool = False, auto_reload: bool = False) -> PolicyManager:
    """Get global policy manager"""
    global _global_policy_manager
    if _global_policy_manager is None or force_reload:
        _global_policy_manager = PolicyManager(auto_reload=auto_reload)
    return _global_policy_manager


# Convenience functions
def get_policy(version: str = None, language: str = "zh") -> PolicyConfig:
    """Quickly get policy configuration"""
    return get_policy_manager().get(version, language)

def get_max_levels_for_intents(intents: Dict[str, float], version: str = None, language: str = "zh") -> Dict[str, str]:
    """Quickly get maximum granularity limits for intent vector"""
    return get_policy_manager().get_max_levels_for_intents(intents, version, language)

def get_policy_json_string(version: str = None, language: str = "zh") -> str:
    """Quickly get policy configuration as JSON string"""
    return get_policy_manager().get_as_json_string(version, language)

def reload_policies(version: str = None, language: str = None):
    """
    Manually reload policy configuration
    
    Args:
        version: Version number, if None reloads all versions
        language: Language identifier, if None and version is not None, reloads all languages for that version
    """
    return get_policy_manager().reload(version, language)


# Usage example
if __name__ == "__main__":
    # Test loading (with auto-reload enabled)
    manager = get_policy_manager(auto_reload=True)
    
    logger.info(f"Supported versions: {manager.list_versions()}")
    logger.info(f"Supported languages: {manager.list_languages()}")
    
    # Get latest version policy configuration (automatically checks and reloads)
    policy = manager.get()  # Don't specify version, use latest version
    logger.info(f"Policy version: {policy.version}")
    logger.info(f"Module: {policy.module}")
    
    # Get policy configuration with specified version and language
    policy = manager.get(version="3.0", language="zh")
    logger.info(f"Policy version: {policy.version}")
    
    # Test intent vector query
    intents = {"I1": 0.6, "I2": 0.4}
    max_levels = manager.get_max_levels_for_intents(intents, version="3.0", language="zh")
    logger.info(f"Maximum granularity limits for intent vector {intents}: {max_levels}")
    
    # Test getting attribute limit for single intent
    level = manager.get_max_level("I1", "OCC", version="3.0", language="zh")
    logger.info(f"Maximum granularity for OCC attribute under I1 intent: {level}")
    
    # Test getting preset
    preset = manager.get_preset("social_default", version="3.0", language="zh")
    logger.info(f"Preset 'social_default': {preset}")
    
    # Test getting policy JSON string
    policy_json = manager.get_as_json_string(version="3.0", language="zh")
    logger.info(f"Policy JSON string length: {len(policy_json)} characters")
    
    # Manually reload specific version and language policy
    manager.reload(version="3.0", language="zh")
    
    # Manually reload all languages for specific version
    manager.reload(version="3.0")
    
    # Manually reload all policies
    manager.reload()
    
    # Use convenience function to reload
    reload_policies(version="3.0", language="zh")  # Reload specific policy
    reload_policies(version="3.0")                  # Reload all languages for version
    reload_policies()                               # Reload all

