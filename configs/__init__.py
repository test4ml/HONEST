"""
HONEST configuration management module.
Provides unified configuration loading with environment variable support.
"""

import os
import yaml
from typing import Dict, Any, Optional


class ConfigManager:
    """Configuration manager."""

    def __init__(self, config_file: str = "configs/default.yaml"):
        """
        Initialize the configuration manager.

        Args:
            config_file: Path to the config file.
        """
        self.config_file = config_file
        self.config = self._load_config()
        self._load_local_overrides()
        self._override_with_env()

    def _load_local_overrides(self):
        """Load git-ignored local override config (configs/local.yaml).

        If present, it is deep-merged over the default config. This is used
        to store sensitive values such as API keys without committing them
        to version control. Environment variables (``_override_with_env``)
        still take the highest priority.
        """
        local_path = os.path.join(os.path.dirname(self.config_file), "local.yaml")
        if not os.path.exists(local_path):
            return
        try:
            with open(local_path, 'r', encoding='utf-8') as f:
                local = yaml.safe_load(f) or {}
            self._deep_update(self.config, local)
        except Exception as e:
            print(f"Warning: failed to load local config {local_path}: {e}")

    @staticmethod
    def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Recursively merge ``override`` into ``base`` (in place)."""
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                ConfigManager._deep_update(base[key], value)
            else:
                base[key] = value

    def _load_config(self) -> Dict[str, Any]:
        """Load the YAML config file."""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_file} not found, using defaults")
            return {}
        except Exception as e:
            print(f"Error loading config file: {e}")
            return {}

    def _override_with_env(self):
        """Override config values with environment variables."""
        # Knowledge graph config
        if os.getenv('MEMGRAPH_URI'):
            self.config['knowledge_graph']['memgraph']['uri'] = os.getenv('MEMGRAPH_URI')
        if os.getenv('MEMGRAPH_USER'):
            self.config['knowledge_graph']['memgraph']['user'] = os.getenv('MEMGRAPH_USER')
        if os.getenv('MEMGRAPH_PASSWORD'):
            self.config['knowledge_graph']['memgraph']['password'] = os.getenv('MEMGRAPH_PASSWORD')

        # LLM config
        self.config.setdefault('llm', {})
        self.config['llm'].setdefault('local', {})
        self.config['llm'].setdefault('deepseek', {})
        self.config['llm'].setdefault('deepseek_v4_flash', {})
        self.config['llm'].setdefault('glm_5_turbo', {})
        self.config['llm'].setdefault('anthropic', {})
        self.config['llm'].setdefault('gemini', {})

        if os.getenv('LLM_PROTOCOL'):
            self.config['llm']['local']['protocol'] = os.getenv('LLM_PROTOCOL')
        if os.getenv('LLM_BASE_URL'):
            self.config['llm']['local']['base_url'] = os.getenv('LLM_BASE_URL')
        if os.getenv('LLM_API_KEY'):
            self.config['llm']['local']['api_key'] = os.getenv('LLM_API_KEY')
        if os.getenv('LLM_MODEL_NAME'):
            self.config['llm']['local']['default_model'] = os.getenv('LLM_MODEL_NAME')

        # DeepSeek config
        if os.getenv('DEEPSEEK_BASE_URL'):
            self.config['llm']['deepseek']['base_url'] = os.getenv('DEEPSEEK_BASE_URL')
            self.config['llm']['deepseek_v4_flash']['base_url'] = os.getenv('DEEPSEEK_BASE_URL')
        if os.getenv('DEEPSEEK_API_KEY'):
            self.config['llm']['deepseek']['api_key'] = os.getenv('DEEPSEEK_API_KEY')
            self.config['llm']['deepseek_v4_flash']['api_key'] = os.getenv('DEEPSEEK_API_KEY')
        if os.getenv('DEEPSEEK_MODEL_NAME'):
            self.config['llm']['deepseek']['default_model'] = os.getenv('DEEPSEEK_MODEL_NAME')
        if os.getenv('DEEPSEEK_V4_FLASH_MODEL_NAME'):
            self.config['llm']['deepseek_v4_flash']['default_model'] = os.getenv('DEEPSEEK_V4_FLASH_MODEL_NAME')

        # GLM config
        if os.getenv('GLM_BASE_URL'):
            self.config['llm']['glm_5_turbo']['base_url'] = os.getenv('GLM_BASE_URL')
        if os.getenv('GLM_API_KEY'):
            self.config['llm']['glm_5_turbo']['api_key'] = os.getenv('GLM_API_KEY')
        if os.getenv('GLM_5_TURBO_MODEL_NAME'):
            self.config['llm']['glm_5_turbo']['default_model'] = os.getenv('GLM_5_TURBO_MODEL_NAME')

        # Anthropic config
        if os.getenv('ANTHROPIC_BASE_URL'):
            self.config['llm']['anthropic']['base_url'] = os.getenv('ANTHROPIC_BASE_URL')
        if os.getenv('ANTHROPIC_API_KEY'):
            self.config['llm']['anthropic']['api_key'] = os.getenv('ANTHROPIC_API_KEY')
        if os.getenv('ANTHROPIC_MODEL_NAME'):
            self.config['llm']['anthropic']['default_model'] = os.getenv('ANTHROPIC_MODEL_NAME')

        # Gemini config
        if os.getenv('GEMINI_API_KEY'):
            self.config['llm']['gemini']['api_key'] = os.getenv('GEMINI_API_KEY')
        if os.getenv('GEMINI_MODEL_NAME'):
            self.config['llm']['gemini']['default_model'] = os.getenv('GEMINI_MODEL_NAME')

        # Data processing config
        if os.getenv('RULE_HEAD_COVERAGE_THRESHOLD'):
            self.config['data_processing']['rule_filtering']['head_coverage'] = float(os.getenv('RULE_HEAD_COVERAGE_THRESHOLD'))
        if os.getenv('RULE_STD_CONFIDENCE_THRESHOLD'):
            self.config['data_processing']['rule_filtering']['std_confidence'] = float(os.getenv('RULE_STD_CONFIDENCE_THRESHOLD'))
        if os.getenv('RULE_PCA_CONFIDENCE_THRESHOLD'):
            self.config['data_processing']['rule_filtering']['pca_confidence'] = float(os.getenv('RULE_PCA_CONFIDENCE_THRESHOLD'))
        if os.getenv('RULE_POSITIVE_EXAMPLES_THRESHOLD'):
            self.config['data_processing']['rule_filtering']['positive_examples'] = int(os.getenv('RULE_POSITIVE_EXAMPLES_THRESHOLD'))

        # Instance matching config
        if os.getenv('MAX_EXAMPLES_PER_RULE'):
            self.config['data_processing']['instance_matching']['max_examples_per_rule'] = int(os.getenv('MAX_EXAMPLES_PER_RULE'))
        if os.getenv('TEST_RULES_LIMIT'):
            self.config['data_processing']['instance_matching']['test_rules_limit'] = int(os.getenv('TEST_RULES_LIMIT'))

        # Mutation config
        if os.getenv('DEFAULT_MUTATION_OPERATORS'):
            operators = os.getenv('DEFAULT_MUTATION_OPERATORS').split(',')
            self.config['mutation']['default_operators'] = [op.strip() for op in operators]
        if os.getenv('MAX_FILES'):
            self.config['mutation']['limits']['max_files'] = int(os.getenv('MAX_FILES')) if os.getenv('MAX_FILES') else None
        if os.getenv('MAX_RULES_PER_FILE'):
            self.config['mutation']['limits']['max_rules_per_file'] = int(os.getenv('MAX_RULES_PER_FILE'))

        # Question generation config
        if os.getenv('QUESTION_TYPES'):
            question_types = os.getenv('QUESTION_TYPES').split(',')
            self.config['question_generation']['question_types'] = [qt.strip() for qt in question_types]
        if os.getenv('MULTIPLE_CHOICE_NUM_OPTIONS'):
            self.config['question_generation']['multiple_choice']['num_choices'] = int(os.getenv('MULTIPLE_CHOICE_NUM_OPTIONS'))

        # Performance config
        if os.getenv('MAX_WORKERS'):
            self.config['performance']['max_workers'] = int(os.getenv('MAX_WORKERS'))
        if os.getenv('BATCH_SIZE'):
            self.config['performance']['batch_size'] = int(os.getenv('BATCH_SIZE'))
        if os.getenv('ENABLE_METADATA_CACHE'):
            self.config['performance']['cache']['enable_metadata_cache'] = os.getenv('ENABLE_METADATA_CACHE').lower() == 'true'
        if os.getenv('ENABLE_FUNCTIONALITY_CACHE'):
            self.config['performance']['cache']['enable_functionality_cache'] = os.getenv('ENABLE_FUNCTIONALITY_CACHE').lower() == 'true'

        # Logging config
        if os.getenv('LOG_LEVEL'):
            self.config['logging']['level'] = os.getenv('LOG_LEVEL')
        if os.getenv('LOG_FORMAT'):
            self.config['logging']['format'] = os.getenv('LOG_FORMAT')

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value."""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_memgraph_config(self) -> Dict[str, str]:
        """Get the Memgraph config."""
        return self.get('knowledge_graph.memgraph', {})

    def get_llm_config(self, provider: str = 'local') -> Dict[str, str]:
        """Get the LLM config."""
        return self.get(f'llm.{provider}', {})

    def get_rule_filtering_config(self) -> Dict[str, Any]:
        """Get the rule filtering config."""
        return self.get('data_processing.rule_filtering', {})

    def get_instance_matching_config(self) -> Dict[str, Any]:
        """Get the instance matching config."""
        return self.get('data_processing.instance_matching', {})

    def get_mutation_config(self) -> Dict[str, Any]:
        """Get the mutation config."""
        return self.get('mutation', {})

    def get_question_generation_config(self) -> Dict[str, Any]:
        """Get the question generation config."""
        return self.get('question_generation', {})

    def get_path_config(self, category: str, subcategory: str = None) -> str:
        """Get a path config."""
        if subcategory:
            return self.get(f'paths.{category}.{subcategory}', '')
        return self.get(f'paths.{category}', '')


# Global config instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """Get a config value (shortcut)."""
    return get_config_manager().get(key, default)
