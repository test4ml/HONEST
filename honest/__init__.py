__version__ = "2.0.0"

# Main API exports - enhanced package structure
# Rule parser - import from new module
from .rule_parser import RuleParser
from .horn_rule_parser import HornRuleParser, AMIERuleParser  # AMIERuleParser for backward compatibility

# Knowledge graph module
from .kg import (
    KnowledgeGraph,
    ComprehensiveKnowledgeGraph,
    MemgraphKnowledgeGraph,
    BaseKnowledgeGraph,
    OptimizedMetadataStore
)

# Question generation module
from .qgen import (
    QuestionGenerator,
    QuestionType,
    PropertyManager,
    GrammarAnalyzer
)

# Rule matching module
from .matcher import PositiveExampleMatcher

# Rule mutation module
from .mutation import (
    MutationOperator,
    FactInstance,
    MutationEngine,
    BodyPermutation,
    BodyAugmentation,
    BodyReduction,
    EntityRename,
    RuleMerging
)

# Language tool module
from .langtool import (
    SentenceAnalyzer,
    NegationTransformer,
    YesNoQuestionTransformer,
    WhTransformer,
    ArticleAnalyzer,
    NumberAnalyzer
)

# Utility module
from .utils.profiling import profile

# Functionality cache functions
from .functionality_cache_loader import (
    load_functionality_cache,
    save_functionality_cache,
    load_functionality_cache_from_json,
    get_cache_info
)

__all__ = [
    # Core classes - rule parsers
    'RuleParser',              # Hybrid logic parser (OR/AND/NOT)
    'HornRuleParser',          # Horn rule parser (AND only)
    'AMIERuleParser',          # Alias for HornRuleParser (backward compatibility)
    'PositiveExampleMatcher',

    # Knowledge graph
    'KnowledgeGraph',
    'ComprehensiveKnowledgeGraph',
    'MemgraphKnowledgeGraph',
    'BaseKnowledgeGraph',
    'OptimizedMetadataStore',

    # Question generation
    'QuestionGenerator',
    'QuestionType',
    'PropertyManager',
    'GrammarAnalyzer',

    # Rule mutation
    'MutationOperator',
    'FactInstance',
    'MutationEngine',
    'BodyPermutation',
    'BodyAugmentation',
    'BodyReduction',
    'EntityRename',
    'RuleMerging',

    # Language tools
    'SentenceAnalyzer',
    'NegationTransformer',
    'YesNoQuestionTransformer',
    'WhTransformer',
    'ArticleAnalyzer',
    'NumberAnalyzer',

    # Utilities
    'profile',

    # Functionality cache functions
    'load_functionality_cache',
    'save_functionality_cache',
    'load_functionality_cache_from_json',
    'get_cache_info'
]