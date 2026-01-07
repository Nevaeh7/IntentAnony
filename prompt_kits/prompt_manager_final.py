"""
Prompt Manager - File-driven version

Features:
 Automatically load YAML configurations from prompts/ folder
 Support JSONSchema validation
 Support Jinja2 templates
 Hot reload (optional)
 Complete decoupling: code and prompts separated
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path
import json
import re
from loguru import logger

try:
    from jinja2 import Environment, FileSystemLoader, Template, meta
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logger.warning("Jinja2 not installed. Install with: pip install jinja2")
    meta = None

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    logger.warning("jsonschema not installed. Install with: pip install jsonschema")

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not installed. Install with: pip install pyyaml")


# JSONSchema for prompt configuration
PROMPT_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "category": {"type": "string"},
        "language": {"type": "string"},
        "template": {"type": "string"},
        "variables": {
            "type": "object",
            "additionalProperties": {"type": "string"}
        },
        "output_schema": {"type": "object"},
        "examples": {
            "type": "array",
            "items": {"type": "object"}
        },
        "metadata": {"type": "object"}
    },
    "required": ["name", "template"]
}


@dataclass
class PromptConfig:
    """Prompt configuration"""
    name: str
    template: str
    system_prompt: str = ""
    description: str = ""
    category: str = "general"
    language: str = "zh"
    variables: Dict[str, str] = field(default_factory=dict)
    output_schema: Optional[Dict] = None
    examples: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[Path] = None
    
    def validate_output(self, output: Any) -> bool:
        """Validate output using JSONSchema"""
        if not JSONSCHEMA_AVAILABLE or not self.output_schema:
            return True
        
        try:
            jsonschema.validate(instance=output, schema=self.output_schema)
            return True
        except jsonschema.exceptions.ValidationError as e:
            logger.error(f"Output validation failed: {e}")
            return False
        


class PromptManager:
    """Prompt Manager - Load from file system"""
    
    def __init__(self, prompts_dir: Optional[Path] = None, auto_reload: bool = True, 
                 default_category: Optional[str] = None, default_language: str = "zh"):
        self.prompts_dir = prompts_dir or Path(__file__).parent / "prompts"
        self.auto_reload = auto_reload
        self._prompts: Dict[str, Dict[str, PromptConfig]] = {}
        self._file_timestamps: Dict[str, float] = {}
        
        # Default category and language
        self._default_category = default_category
        self._default_language = default_language
        
        # Ensure directory exists
        self.prompts_dir.mkdir(exist_ok=True)
        
        # Initialize Jinja2 environment
        if JINJA2_AVAILABLE:
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.prompts_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True
            )
        else:
            self.jinja_env = None
            logger.warning("Jinja2 not available, will use simple string formatting")
        
        # Load all prompts
        self.load_all()
        self.set_defaults(default_category, default_language)
    
    def load_all(self):
        """Load all configuration files (supports multiple formats)"""
        # Supported file formats
        file_patterns = [
            "*.yaml",
            "*.yml", 
            "*.json"  # Future extension to .toml
        ]
        
        # Collect all files
        prompt_files = []
        for pattern in file_patterns:
            prompt_files.extend(list(self.prompts_dir.glob(pattern)))
        
        # Load files
        for file_path in prompt_files:
            try:
                self.load_from_file(file_path)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(self._prompts)} categories of prompts (total {len(prompt_files)} files)")
    
    def load_from_file(self, file_path: Path) -> PromptConfig:
        """Load prompt configuration from file (supports multiple formats)"""
        # Select loading method based on file extension
        suffix = file_path.suffix.lower()
        
        if suffix in ['.yaml', '.yml']:
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML not installed")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        elif suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        # Validate configuration
        if JSONSCHEMA_AVAILABLE:
            try:
                jsonschema.validate(instance=data, schema=PROMPT_CONFIG_SCHEMA)
            except jsonschema.exceptions.ValidationError as e:
                logger.error(f"Configuration validation failed {file_path}: {e}")
        
        # Create configuration object
        config = PromptConfig(
            name=data.get('name', 'Unnamed'),
            template=data.get('template', ''),
            system_prompt=data.get('system_prompt', ''),
            description=data.get('description', ''),
            category=data.get('category', 'general'),
            language=data.get('language', 'zh'),
            variables=data.get('variables', {}),
            output_schema=data.get('output_schema'),
            examples=data.get('examples', []),
            metadata=data.get('metadata', {}),
            file_path=file_path
        )
        
        # Register prompt
        self.register(config.category, config.language, config)
        
        # Record file timestamp
        self._file_timestamps[str(file_path)] = file_path.stat().st_mtime
        
        return config
    
    def register(self, category: str, language: str, config: PromptConfig):
        """Register prompt configuration"""
        if category not in self._prompts:
            self._prompts[category] = {}
        
        self._prompts[category][language] = config
        # logger.debug(f"Registered prompt: {category}_{language}")
    
    def check_reload(self, category: str, language: str = "zh"):
        """Check and reload file (if auto-reload is enabled)
        
        Args:
            category: Category name
            language: Language identifier
        """
        if not self.auto_reload:
            return
        
        config = self._prompts.get(category, {}).get(language)
        if not config or not config.file_path:
            return
        
        file_path = config.file_path
        current_mtime = file_path.stat().st_mtime
        cached_mtime = self._file_timestamps.get(str(file_path), 0)
        
        if current_mtime > cached_mtime:
            logger.info(f"File change detected, reloading: {file_path}")
            self.load_from_file(file_path)
    
    def set_defaults(self, category: Optional[str] = None, language: Optional[str] = None):
        """Set default category and language"""
        if category is not None:
            self._default_category = category
        if language is not None:
            self._default_language = language
        logger.debug(f"Set defaults: category={self._default_category}, language={self._default_language}")
    
    def get_defaults(self) -> Tuple[Optional[str], str]:
        """Get current default values"""
        return self._default_category, self._default_language
    
    def get(self, category: Optional[str] = None, language: Optional[str] = None, **kwargs) -> str:
        """Get and render prompt
        
        Args:
            category: Category name, if None uses default value
            language: Language identifier, if None uses default value
            **kwargs: Variables to pass to template
        """
        # Use default values
        category = category or self._default_category
        language = language or self._default_language
        
        if category is None:
            raise ValueError("category must be provided, or set default via set_defaults()")
        
        # Check reload
        self.check_reload(category, language)
        
        if category not in self._prompts:
            raise ValueError(f"Category not found: {category}")
        
        if language not in self._prompts[category]:
            raise ValueError(f"Language not found: {language} (category: {category})")
        
        config = self._prompts[category][language]
        template_str = config.template
        
        # Use Jinja2 rendering (if available)
        if JINJA2_AVAILABLE and self.jinja_env:
            try:
                template = self.jinja_env.from_string(template_str)
                return template.render(**kwargs)
            except Exception as e:
                logger.error(f"Jinja2 rendering failed: {e}")
                # Jinja2 syntax errors should not fall back to .format(), as .format() cannot handle Jinja2 syntax
                # Raise exception so user can fix template issues
                raise ValueError(f"Template rendering failed (possibly Jinja2 syntax error): {e}") from e
        else:
            # If Jinja2 not available, try simple string formatting
            # Note: This will fail if template contains Jinja2 syntax
            try:
                return template_str.format(**kwargs) if kwargs else template_str
            except KeyError as e:
                # If template contains Jinja2 syntax but Jinja2 unavailable, provide hint
                logger.error(f"String formatting failed, template may use Jinja2 syntax but Jinja2 unavailable: {e}")
                raise ValueError(
                    f"Template contains unresolved syntax, please install Jinja2: pip install jinja2"
                ) from e
    

    def get_config(self, category: Optional[str] = None, language: Optional[str] = None) -> PromptConfig:
        """Get prompt configuration object
        
        Args:
            category: Category name, if None uses default value
            language: Language identifier, if None uses default value
        """
        # Use default values
        category = category or self._default_category
        language = language or self._default_language
        
        if category is None:
            raise ValueError("category must be provided, or set default via set_defaults()")
        
        self.check_reload(category, language)
        
        if category not in self._prompts:
            raise ValueError(f"Category not found: {category}")
        if language not in self._prompts[category]:
            raise ValueError(f"Language not found: {language} (category: {category})")
        return self._prompts[category][language]
    
    def validate_output(self, output: Any, category: Optional[str] = None, language: Optional[str] = None) -> bool:
        """Validate if LLM output matches schema
        
        Args:
            output: Output to validate
            category: Category name, if None uses default value
            language: Language identifier, if None uses default value
        """
        config = self.get_config(category, language)
        return config.validate_output(output)
    
    def list_categories(self) -> List[str]:
        """List all categories"""
        return list(self._prompts.keys())
    
    def list_languages(self, category: str) -> List[str]:
        """List all languages for a category"""
        if category not in self._prompts:
            return []
        return list(self._prompts[category].keys())
    
    def list_all_prompts(self) -> Dict[str, List[str]]:
        """List all prompts"""
        return {
            category: list(languages.keys())
            for category, languages in self._prompts.items()
        }
    def get_system_prompt(self, category: Optional[str] = None, language: Optional[str] = None) -> str:
        """Get system prompt, returns empty string if system prompt does not exist
        
        Args:
            category: Category name, if None uses default value
            language: Language identifier, if None uses default value
        """
        config = self.get_config(category, language)
        return config.system_prompt

    def get_messages(self, category: Optional[str] = None, language: Optional[str] = None, **kwargs) -> list:
        """Get message list (contains system and user messages)
        
        Args:
            category: Category name, if None uses default value
            language: Language identifier, if None uses default value
            **kwargs: Variables to pass to template
        
        Returns:
            Message list, format: [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        """
        messages = []
        system_prompt = self.get_system_prompt(category, language)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        prompt = self.get(category, language, **kwargs)
        messages.append({"role": "user", "content": prompt})
        return messages
    
    def get_messages_for_anonymized(self, category: Optional[str] = None, language: Optional[str] = None, user_prompt: str = None, **kwargs) -> list:
        """Get message list (contains system and user messages)
        
        Args:
            category: Category name, if None uses default value
            language: Language identifier, if None uses default value
            **kwargs: Variables to pass to template
        """
        anony_prompt = self.get(category, language, **kwargs)
        messages = [
                {"role": "system", "content": anony_prompt},
                {"role": "user", "content": user_prompt}
        ]
        # logger.info(f"anonymized prompt: {messages}")
        return messages

    
    def bind(self, category: str, language: str = "zh"):
        """Bind category and language, return a convenience function that doesn't require passing these two parameters each time
        
        Args:
            category: Category name
            language: Language identifier, defaults to "zh"
        
        Returns:
            A function that only needs template variables
        
        Example:
            >>> manager = PromptManager()
            >>> get_utility_messages = manager.bind("eval_utility", "en")
            >>> messages = get_utility_messages(original_string="...", latest_string="...")
        """
        def bound_get_messages(**kwargs):
            return self.get_messages(category, language, **kwargs)
        
        # Also provide convenient attribute access
        bound_get_messages.category = category
        bound_get_messages.language = language
        bound_get_messages.get_prompt = lambda **kwargs: self.get(category, language, **kwargs)
        bound_get_messages.get_system_prompt = lambda: self.get_system_prompt(category, language)
        bound_get_messages.get_config = lambda: self.get_config(category, language)
        
        return bound_get_messages
    
    def analyze_template_variables(self, category: Optional[str] = None, language: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Analyze template variables, identify variables required by template and extra variables provided
        
        Args:
            category: Category name, if None uses default value
            language: Language identifier, if None uses default value
            **kwargs: Variables provided for comparison analysis
        
        Returns:
            Dictionary containing the following information:
            - required_vars: Set of variables required by template
            - provided_vars: Set of variables provided
            - missing_vars: Missing variables (required by template but not provided)
            - extra_vars: Extra variables (provided but not needed by template)
            - used_vars: Used variables (required by template and provided)
        """
        # Use default values
        category = category or self._default_category
        language = language or self._default_language
        
        if category is None:
            raise ValueError("category must be provided, or set default via set_defaults()")
        
        config = self.get_config(category, language)
        template_str = config.template
        
        # Analyze template variables
        required_vars = set()
        
        if JINJA2_AVAILABLE and meta:
            try:
                # Parse template, find all undefined variables
                ast = self.jinja_env.parse(template_str)
                required_vars = meta.find_undeclared_variables(ast)
            except Exception as e:
                logger.warning(f"Unable to parse Jinja2 template variables: {e}")
                # If parsing fails, try simple regex matching
                # Match variables in {{ variable }} format
                matches1 = re.findall(r'\{\{[\s]*([a-zA-Z_][a-zA-Z0-9_]*)[\s]*\}\}', template_str)
                # Match variables in {% if variable %} format (excluding keywords)
                matches2 = re.findall(r'\{%[\s]*if[\s]+([a-zA-Z_][a-zA-Z0-9_]*)[\s]*%\}', template_str)
                required_vars = set(matches1 + matches2)
        else:
            # If Jinja2 not available, use simple regex
            # Match {{ variable }} format
            matches1 = re.findall(r'\{\{[\s]*([a-zA-Z_][a-zA-Z0-9_]*)[\s]*\}\}', template_str)
            # Match variables in {% if variable %} format (excluding keywords)
            matches2 = re.findall(r'\{%[\s]*if[\s]+([a-zA-Z_][a-zA-Z0-9_]*)[\s]*%\}', template_str)
            required_vars = set(matches1 + matches2)
        
        # Compare and analyze
        provided_vars = set(kwargs.keys())
        missing_vars = required_vars - provided_vars
        extra_vars = provided_vars - required_vars
        used_vars = required_vars & provided_vars
        
        result = {
            "required_vars": required_vars,
            "provided_vars": provided_vars,
            "missing_vars": missing_vars,
            "extra_vars": extra_vars,
            "used_vars": used_vars,
            "template_source": f"{category}_{language}"
        }
        
        return result
    
    def print_template_analysis(self, category: Optional[str] = None, language: Optional[str] = None, **kwargs):
        """Print template variable analysis results (formatted output)
        
        Args:
            category: Category name, if None uses default value
            language: Language identifier, if None uses default value
            **kwargs: Variables provided for comparison analysis
        
        Example:
            >>> manager.print_template_analysis("eval_utility", "en", 
            ...     original_string="...", latest_string="...", extra_var="...")
        """
        analysis = self.analyze_template_variables(category, language, **kwargs)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Template variable analysis: {analysis['template_source']}")
        logger.info(f"{'='*60}")
        
        logger.info(f"\n📋 Variables required by template ({len(analysis['required_vars'])}):")
        if analysis['required_vars']:
            for var in sorted(analysis['required_vars']):
                status = " Provided" if var in analysis['used_vars'] else " Missing"
                logger.info(f"  - {var} {status}")
        else:
            logger.info("  (None)")
        
        logger.info(f"\n Variables used ({len(analysis['used_vars'])}):")
        if analysis['used_vars']:
            for var in sorted(analysis['used_vars']):
                logger.info(f"  - {var}")
        else:
            logger.info("  (None)")
        
        logger.info(f"\n Missing variables ({len(analysis['missing_vars'])}):")
        if analysis['missing_vars']:
            for var in sorted(analysis['missing_vars']):
                logger.info(f"  - {var}")
        else:
            logger.info("  (None)")
        
        logger.info(f"\n  Extra variables ({len(analysis['extra_vars'])}):")
        if analysis['extra_vars']:
            for var in sorted(analysis['extra_vars']):
                logger.info(f"  - {var} = {repr(kwargs[var])[:50]}")
        else:
            logger.info("  (None)")
        
        logger.info(f"\n{'='*60}\n")

    def reload(self, category: str = None, language: str = None):
        """
        Manually reload prompts
        
        Args:
            category: Category name
                - If both category and language provided: reload specific prompt
                - If only category provided: reload all languages for that category
                - If neither provided: reload all prompts
            language: Language identifier
        """
        if category and language:
            # Reload specific prompt
            config = self._prompts.get(category, {}).get(language)
            if config and config.file_path:
                logger.info(f"Reloading prompt: {category}_{language}")
                self.load_from_file(config.file_path)
            else:
                logger.warning(f"Prompt not found: {category}_{language}")
        elif category:
            # Reload all languages for specific category
            languages = self._prompts.get(category, {})
            if languages:
                logger.info(f"Reloading all languages for category {category}")
                for lang, config in languages.items():
                    if config and config.file_path:
                        self.load_from_file(config.file_path)
            else:
                logger.warning(f"Category not found: {category}")
        else:
            # Reload all
            logger.info("Reloading all prompts")
            self._prompts.clear()
            self._file_timestamps.clear()
            self.load_all()
    
    def create_from_template(self, category: str, language: str, name: str, 
                           description: str, template: str, system_prompt: str, **kwargs):
        """Create new prompt and save to file"""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed")
        
        # Prepare data
        data = {
            "name": name,
            "description": description,
            "category": category,
            "language": language,
            "template": template,
            "system_prompt": system_prompt,
            **kwargs
        }
        
        # Generate filename
        filename = f"{category}_{language}.yaml"
        filepath = self.prompts_dir / filename
        
        # Save file
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        
        logger.info(f"Created prompt file: {filepath}")
        
        # Load new file
        self.load_from_file(filepath)
        
        return filepath


# Global instance
_global_manager = None

def get_manager(force_reload: bool = False, auto_reload: bool = False, 
                default_category: Optional[str] = None, default_language: str = "zh") -> PromptManager:
    """Get global prompt manager
    
    Args:
        force_reload: Whether to force reload
        auto_reload: Whether to enable auto-reload
        default_category: Default category name
        default_language: Default language identifier
    """
    global _global_manager
    if _global_manager is None or force_reload:
        _global_manager = PromptManager(auto_reload=auto_reload, 
                                       default_category=default_category,
                                       default_language=default_language)
    elif default_category is not None or default_language != "zh":
        # If new default values provided, update existing manager's defaults
        _global_manager.set_defaults(default_category, default_language)
    return _global_manager


# Convenience functions
def get_prompt(category: Optional[str] = None, language: Optional[str] = None, **kwargs) -> str:
    """Quickly get prompt
    
    Args:
        category: Category name, if None uses default value
        language: Language identifier, if None uses default value
        **kwargs: Variables to pass to template
    """
    return get_manager().get(category, language, **kwargs)

def set_defaults(category: Optional[str] = None, language: Optional[str] = None):
    """Set default values for global prompt manager
    
    Args:
        category: Default category name
        language: Default language identifier
    """
    get_manager().set_defaults(category, language)

def reload_prompts(category: str = None, language: str = None):
    """
    Manually reload prompts
    
    Args:
        category: Category name, if None reloads all
        language: Language identifier, if None and category is not None, reloads all languages for that category
    """
    return get_manager().reload(category, language)


def bind_prompt(category: str, language: str = "zh"):
    """Bind category and language, return a convenience function
    
    Args:
        category: Category name
        language: Language identifier, defaults to "zh"
    
    Returns:
        A function that only needs template variables
    
    Example:
        >>> get_utility_messages = bind_prompt("eval_utility", "en")
        >>> messages = get_utility_messages(original_string="...", latest_string="...")
    """
    return get_manager().bind(category, language)


def analyze_template(category: Optional[str] = None, language: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Convenience function to analyze template variables
    
    Args:
        category: Category name, if None uses default value
        language: Language identifier, if None uses default value
        **kwargs: Variables provided for comparison analysis
    
    Returns:
        Analysis result dictionary
    """
    return get_manager().analyze_template_variables(category, language, **kwargs)


def print_template_analysis(category: Optional[str] = None, language: Optional[str] = None, **kwargs):
    """Convenience function to print template variable analysis
    
    Args:
        category: Category name, if None uses default value
        language: Language identifier, if None uses default value
        **kwargs: Variables provided for comparison analysis
    """
    get_manager().print_template_analysis(category, language, **kwargs)


# Usage example
if __name__ == "__main__":
    # Test loading (with auto-reload enabled)
    manager = get_manager(auto_reload=True)
    
    logger.info(f"Available categories: {manager.list_categories()}")
    logger.info(f"All prompts: {manager.list_all_prompts()}")
    
    # Get prompt (automatically checks and reloads)
    prompt = manager.get("infer", "en", context="I am a 27-year-old construction engineer")
    logger.info(prompt)
    
    # Manually reload specific prompt
    manager.reload("infer", "en")
    
    # Manually reload all languages for specific category
    manager.reload("infer")
    
    # Manually reload all prompts
    manager.reload()
    
    # Use convenience function to reload
    reload_prompts("infer", "en")  # Reload specific prompt
    reload_prompts("infer")        # Reload all languages for category
    reload_prompts()               # Reload all
    
    # Use bind method to avoid passing category and language each time
    get_utility_messages = manager.bind("eval_utility", "en")
    messages = get_utility_messages(
        original_string="I'm a 28-year-old software engineer",
        latest_string="I'm a software engineer"
    )
    logger.info(f"Bound messages: {messages}")
    
    # Use global convenience function
    get_eval_messages = bind_prompt("eval_utility", "en")
    messages = get_eval_messages(original_string="...", latest_string="...")
    
    # Analyze template variables, identify extra kwargs
    manager.print_template_analysis(
        "eval_utility", "en",
        original_string="Original text here",
        latest_string="Adapted text here",
        extra_unused_var="This will be flagged as extra"
    )
    
    # Or get analysis result (for programmatic use)
    analysis = manager.analyze_template_variables(
        "eval_utility", "en",
        original_string="...",
        latest_string="...",
        unused_var="..."
    )
    print(f"Extra variables: {analysis['extra_vars']}")
