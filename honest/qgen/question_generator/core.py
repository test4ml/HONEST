#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Natural language question generator - Core module"""
from typing import List, Tuple, Dict, Optional, Union, Any, Protocol
import hashlib
import spacy
import logging

logger = logging.getLogger(__name__)

from honest.constants import SPACY_MODEL_NAME
from ..types import QuestionType
from ..grammar_analyzer import GrammarAnalyzer
# Use modules from langtool to replace modules in qgen
from ...langtool import LanguageFactory
from ...langtool.core.types import Language
from ..property_formatter import wikidata_formatter
from ..relation_negation_config import relation_negation_manager, inverse_relation_registry
from ...rule_parser import LogicExpression, LogicOperator, Fact
from ...horn_rule_parser import HornRuleParser
from ..semantic_validator import semantic_validator
from ..distractor_validator import DistractorValidator, DistractorTripleGenerator
from .distractor_engine import DistractorEngine

# Type aliases
TriplePattern = Tuple[str, str, str]
BodyPatterns = List[TriplePattern]
Rule = Union[str, Tuple[BodyPatterns, TriplePattern]]
EntityLabels = Dict[str, str]
RelationLabels = Dict[str, str]
QuestionResult = Dict[str, Any]


class KnowledgeGraphProtocol(Protocol):
    """Knowledge graph interface protocol."""
    def get_entity_label(self, entity_id: str) -> Optional[str]:
        """Get an entity label."""
        ...

    def get_property_label(self, property_id: str) -> Optional[str]:
        """Get a property label."""
        ...


class QuestionGenerator:
    """Natural language question generator."""

    def __init__(self, knowledge_graph: KnowledgeGraphProtocol) -> None:
        """
        Initialize the question generator.

        Args:
            knowledge_graph: Knowledge graph interface; must implement the
                get_entity_label and get_property_label methods.
        """
        self.kg: KnowledgeGraphProtocol = knowledge_graph
        self.entity_cache: Dict[str, str] = {}
        self.property_cache: Dict[str, str] = {}
        self.grammar_analyzer: GrammarAnalyzer = GrammarAnalyzer()

        # Use the langtool factory to create transformers
        self.negation_transformer = LanguageFactory.get_negation_transformer(Language.ENGLISH)
        self.wh_transformer = LanguageFactory.get_wh_transformer(Language.ENGLISH)
        self.yesno_transformer = LanguageFactory.get_yesno_question_transformer(Language.ENGLISH)
        self.article_analyzer = LanguageFactory.get_article_analyzer(Language.ENGLISH)

        # Load the spaCy English model for NER
        try:
            self.nlp = spacy.load(SPACY_MODEL_NAME)
        except OSError:
            print(f"Warning: spaCy English model '{SPACY_MODEL_NAME}' not found.")
            print(f"Install it with: python -m spacy download {SPACY_MODEL_NAME}")
            self.nlp = None

        # Initialize the distractor validator (ensures correctness of MCQ options).
        # If the KG supports the contains method, the CWA is used to verify that
        # distractors are indeed false.
        self.distractor_validator = DistractorValidator(knowledge_graph)
        self.distractor_triple_generator = DistractorTripleGenerator(knowledge_graph)

        # Initialize the distractor generation engine
        self.distractor_engine = DistractorEngine(self)

    # Backward compatibility: delegate methods to distractor_engine
    def _extract_augmented_features(self, *args, **kwargs):
        """Delegate to distractor_engine for backward compatibility"""
        return self.distractor_engine.extract_augmented_features(*args, **kwargs)

    def _is_duplicate_of_premise(self, *args, **kwargs):
        """Delegate to distractor_engine for backward compatibility"""
        return self.distractor_engine.is_duplicate_of_premise(*args, **kwargs)

    def _get_all_premise_statements(self, *args, **kwargs):
        """Delegate to distractor_engine for backward compatibility"""
        return self.distractor_engine.get_all_premise_statements(*args, **kwargs)

    def _generate_distractors_deterministic(self, *args, **kwargs):
        """Delegate to distractor_engine for backward compatibility"""
        return self.distractor_engine.generate_distractors_deterministic(*args, **kwargs)

    def generate_questions(self,
                         rule: Rule,
                         instantiation: Dict[str, str],
                         question_types: Optional[List[QuestionType]] = None,
                         focus_entities: Optional[List[str]] = None,
                         focus_relations: Optional[List[str]] = None,
                         num_choices: int = 4,
                         answer_position_seed: Optional[int] = None) -> List[QuestionResult]:
        """
        Generate natural language questions from an inference rule.

        Args:
            rule: Inference rule; can be a string or a parsed format
            instantiation: Mapping from variables to entities
            question_types: List of question types to generate
            focus_entities: List of entities to focus on
            focus_relations: List of relations to focus on
            num_choices: Number of options in multiple choice questions
            answer_position_seed: Optional seed for deterministic answer position in MCQs.
                                 If provided, ensures consistent answer positions across
                                 original and mutated question pairs (for metamorphic testing).
                                 If None, uses hash-based method (backward compatibility).

        Returns:
            List[Dict]: List of generated questions; each question contains the type,
            content, answer, and other information.
        """
        # Parse the rule.
        # Handle both string format and tuple format
        if isinstance(rule, str):
            parsed_rule = HornRuleParser.parse_rule(rule)
            body_patterns = [fact.to_tuple() for fact in parsed_rule.body]
            head_pattern = parsed_rule.head.to_tuple()
        elif isinstance(rule, tuple) and len(rule) == 2:
            # Tuple format: ([body_patterns], head_pattern)
            body_patterns, head_pattern = rule
        else:
            raise ValueError(f"Invalid rule format: {rule}")

        # CRITICAL FIX: Instantiate patterns by replacing variables with actual entity IDs
        # Before: body_patterns contains ('?a', 'P123', '?b')
        # After: body_patterns contains ('Q456', 'P123', 'Q789')
        # This is essential for:
        # 1. Distractor generation - generators need actual entity IDs, not variable names
        # 2. KG validation - validator needs to query KG with real entities
        # 3. Text conversion - formatter needs entity IDs to look up labels
        def instantiate_pattern(pattern: TriplePattern) -> TriplePattern:
            """Replace variables in pattern with actual entity IDs from instantiation"""
            s, p, o = pattern
            # Clean variable names (remove '?' prefix if present)
            clean_s = self._clean_variable_name(s)
            clean_o = self._clean_variable_name(o)
            # Look up in instantiation mapping
            actual_s = instantiation.get(s, instantiation.get(clean_s, s))
            actual_o = instantiation.get(o, instantiation.get(clean_o, o))
            return (actual_s, p, actual_o)

        # Instantiate all patterns
        body_patterns = [instantiate_pattern(bp) for bp in body_patterns]
        head_pattern = instantiate_pattern(head_pattern)

        # By default, generate all question types
        if question_types is None:
            question_types = list(QuestionType)

        # Retrieve labels for entities and relations.
        # Handle both List[TriplePattern] and LogicExpression
        if isinstance(body_patterns, LogicExpression):
            # Extract all facts from LogicExpression for label extraction
            all_facts = self._extract_all_facts(body_patterns)
            all_patterns = [fact.to_tuple() for fact in all_facts] + [head_pattern]
        else:
            all_patterns = body_patterns + [head_pattern]

        entity_labels = self._get_entity_labels(instantiation, all_patterns)
        relation_labels = self._get_relation_labels(all_patterns)

        questions = []

        for question_type in question_types:
            question = None
            if question_type == QuestionType.YES_NO:
                question = self._generate_yes_no_question(
                    body_patterns, head_pattern, entity_labels, relation_labels,
                    focus_entities, focus_relations
                )
            elif question_type == QuestionType.WH_QUESTION:
                question = self._generate_wh_question(
                    body_patterns, head_pattern, entity_labels, relation_labels,
                    focus_entities, focus_relations
                )
            elif question_type == QuestionType.TRUE_FALSE:
                question = self._generate_true_false_question(
                    body_patterns, head_pattern, entity_labels, relation_labels,
                    focus_entities, focus_relations
                )
            elif question_type == QuestionType.MULTIPLE_CHOICE:
                question = self._generate_multiple_choice_question(
                    body_patterns, head_pattern, entity_labels, relation_labels,
                    focus_entities, focus_relations, num_choices, answer_position_seed
                )

            if question:
                questions.append(question)

        return questions

    def _generate_yes_no_question(self,
                                 body_patterns: BodyPatterns,
                                 head_pattern: TriplePattern,
                                 entity_labels: EntityLabels,
                                 relation_labels: RelationLabels,
                                 focus_entities: Optional[List[str]],
                                 focus_relations: Optional[List[str]]) -> QuestionResult:
        """Generate a yes/no question."""
        premises = self._format_premises(body_patterns, entity_labels, relation_labels)
        conclusion = self._format_conclusion(head_pattern, entity_labels, relation_labels)

        # Directly construct a yes/no question, preserving the original format
        question_text = f"Given that {premises}, is it true that {conclusion}?"

        return {
            'type': QuestionType.YES_NO.value,
            'question': question_text,
            'answer': "Yes, this can be logically inferred from the premises.",
            'correct_answer': True,
            'reasoning': self._generate_reasoning(body_patterns, head_pattern, entity_labels, relation_labels)
        }

    def _generate_wh_question(self,
                             body_patterns: BodyPatterns,
                             head_pattern: TriplePattern,
                             entity_labels: EntityLabels,
                             relation_labels: RelationLabels,
                             focus_entities: Optional[List[str]],
                             focus_relations: Optional[List[str]]) -> QuestionResult:
        """Generate a WH-question using a specialized grammar transformer."""
        premises = self._format_premises(body_patterns, entity_labels, relation_labels)

        # Get the subject, predicate, and object from the head pattern
        head_subject, head_predicate, head_object = head_pattern

        # Query the functionality and inverse functionality of the relation
        functionality = self._get_functionality(head_predicate)
        inverse_functionality = self._get_inverse_functionality(head_predicate)

        # Choose the direction with greater functionality as the question focus
        if functionality >= inverse_functionality:
            # Higher functionality: ask for the object (the s->o direction has
            # higher functionality, so ask about o)
            question_entity = head_subject
            question_entity_label = entity_labels.get(question_entity, question_entity)
            answer_entity = head_object
            answer_entity_label = entity_labels.get(answer_entity, answer_entity)
            relation_label = relation_labels.get(head_predicate, head_predicate)

            # Build a correct declarative statement
            statement = self._format_statement(question_entity_label, relation_label, answer_entity_label, head_predicate)

            # Select an appropriate question word based on the statement's
            # semantics and the answer entity type
            question_word = self._select_question_word_from_statement(statement, answer_entity_label, head_predicate, relation_label)

            # FIX: Remove parenthetical descriptions from statement before passing to wh_transformer
            # to avoid spaCy misidentifying subject/verb from clauses inside parentheses
            statement_for_parsing = self._extract_pure_entities_from_statement(statement)
            answer_entity_pure = self.article_analyzer._extract_pure_label(answer_entity_label)

            # Use the grammar transformer to generate the question
            question_text = self.wh_transformer.transform(
                statement_for_parsing,
                question_word=question_word,
                focus_entity=answer_entity_pure
            )

            # Prepend premise information; ensure the question starts with a lowercase letter
            if premises:
                # Lowercase the question's first letter, since it is no longer at the start of a sentence
                question_text = question_text[0].lower() + question_text[1:] if question_text else question_text
                question_text = f"Given that {premises}, {question_text}"
                # Normalize Template and Category formats
                question_text = self._normalize_template_category_names(question_text)

            answer = f"{answer_entity_label}."
        else:
            # Higher inverse functionality: ask for the subject (the o->s
            # direction has higher functionality, so ask about s)
            question_entity = head_object
            question_entity_label = entity_labels.get(question_entity, question_entity)
            answer_entity = head_subject
            answer_entity_label = entity_labels.get(answer_entity, answer_entity)
            relation_label = relation_labels.get(head_predicate, head_predicate)

            # Build a correct declarative statement: answer_entity relation question_entity
            statement = self._format_statement(answer_entity_label, relation_label, question_entity_label, head_predicate)

            # Select an appropriate question word based on the statement's
            # semantics and the answer entity type
            question_word = self._select_question_word_from_statement(statement, answer_entity_label, head_predicate, relation_label)

            # FIX: Remove parenthetical descriptions from statement before passing to wh_transformer
            # to avoid spaCy misidentifying subject/verb from clauses inside parentheses
            statement_for_parsing = self._extract_pure_entities_from_statement(statement)
            answer_entity_pure = self.article_analyzer._extract_pure_label(answer_entity_label)

            # Use the grammar transformer to generate the question
            question_text = self.wh_transformer.transform(
                statement_for_parsing,
                question_word=question_word,
                focus_entity=answer_entity_pure
            )

            # Prepend premise information; ensure the question starts with a lowercase letter
            if premises:
                # Lowercase the question's first letter, since it is no longer at the start of a sentence
                question_text = question_text[0].lower() + question_text[1:] if question_text else question_text
                question_text = f"Given that {premises}, {question_text}"
                # Normalize Template and Category formats
                question_text = self._normalize_template_category_names(question_text)

            answer = f"{answer_entity_label}."

        return {
            'type': QuestionType.WH_QUESTION.value,
            'question': question_text,
            'answer': answer,
            'correct_answer': answer_entity_label,
            'reasoning': self._generate_reasoning(body_patterns, head_pattern, entity_labels, relation_labels),
            'functionality_info': {
                'predicate': head_predicate,
                'functionality': functionality,
                'inverse_functionality': inverse_functionality,
                'question_direction': 'subject_to_object' if functionality >= inverse_functionality else 'object_to_subject'
            }
        }

    def _generate_true_false_question(self,
                                     body_patterns: BodyPatterns,
                                     head_pattern: TriplePattern,
                                     entity_labels: EntityLabels,
                                     relation_labels: RelationLabels,
                                     focus_entities: Optional[List[str]],
                                     focus_relations: Optional[List[str]]) -> QuestionResult:
        """Generate a true/false question."""
        premises = self._format_premises(body_patterns, entity_labels, relation_labels)
        conclusion = self._format_conclusion(head_pattern, entity_labels, relation_labels)

        # Use the formatted conclusion directly, preserving the original format.
        # Handle both List[TriplePattern] and LogicExpression
        num_premises = self._count_premises(body_patterns)
        if num_premises == 1:
            question_text = f"Based on the premise that {premises}, is it true that {conclusion}?"
        else:
            question_text = f"Based on the premises that {premises}, is it true that {conclusion}?"

        return {
            'type': QuestionType.TRUE_FALSE.value,
            'question': question_text,
            'answer': "True, this conclusion logically follows from the premises.",
            'correct_answer': True,
            'reasoning': self._generate_reasoning(body_patterns, head_pattern, entity_labels, relation_labels)
        }

    def _generate_multiple_choice_question(self,
                                          body_patterns: BodyPatterns,
                                          head_pattern: TriplePattern,
                                          entity_labels: EntityLabels,
                                          relation_labels: RelationLabels,
                                          focus_entities: Optional[List[str]],
                                          focus_relations: Optional[List[str]],
                                          num_choices: int,
                                          answer_position_seed: Optional[int] = None) -> QuestionResult:
        """Generate a multiple choice question.

        CRITICAL FIX: Answer position consistency for metamorphic testing

        If answer_position_seed is provided:
        - Uses seed to determine answer position
        - Ensures original and mutated questions have the same answer position
        - Required for proper metamorphic testing

        If answer_position_seed is None (backward compatibility):
        - Uses head pattern hash to determine answer position
        - Maintains existing behavior for code that doesn't pass a seed

        The correct answer returns the option letter (A/B/C/D) rather than the
        full option text.
        """
        premises = self._format_premises(body_patterns, entity_labels, relation_labels)
        correct_conclusion = self._format_conclusion(head_pattern, entity_labels, relation_labels)

        # Generate distractor options - delegated to DistractorEngine
        distractors = self.distractor_engine.generate_distractors_deterministic(
            body_patterns, head_pattern, entity_labels,
            relation_labels, num_choices - 1
        )

        # Combine options
        options = []

        # CRITICAL FIX: Deterministic answer position for metamorphic testing
        #
        # If answer_position_seed is provided (NEW):
        #   - Use the seed to determine correct answer position
        #   - Ensures original and mutated questions have the SAME answer position
        #   - Required for proper metamorphic testing (original and mutated should be consistent)
        #
        # If answer_position_seed is None (backward compatibility):
        #   - Use hash-based method (old behavior)
        #   - Hash of head_pattern determines position
        #
        # Note: The seed-based approach is superior because:
        # 1. Original and mutated instances can share the same seed
        # 2. Seed is stored in output CSV for full reproducibility
        # 3. No dependency on head_pattern structure (avoids variable name sensitivity)

        if answer_position_seed is not None:
            # NEW: Seed-based position (for metamorphic testing consistency)
            correct_index = answer_position_seed % num_choices
            logger.debug(
                f"Using provided seed {answer_position_seed} for answer position → "
                f"position {correct_index} (letter {chr(65 + correct_index)})"
            )
        else:
            # OLD: Hash-based position (backward compatibility)
            head_pattern_str = str(head_pattern)
            correct_answer_hash = hashlib.md5(head_pattern_str.encode('utf-8')).hexdigest()
            seed_value = int(correct_answer_hash[:8], 16)
            correct_index = seed_value % num_choices
            logger.debug(
                f"Using hash-based position (no seed provided) → "
                f"position {correct_index} (letter {chr(65 + correct_index)})"
            )

        # Verify that the number of distractors is sufficient
        required_distractors = num_choices - 1
        if len(distractors) < required_distractors:
            logger.warning(
                f"Insufficient distractors: {len(distractors)} generated, "
                f"{required_distractors} required. Skipping multiple choice question."
            )
            return None

        # Fill the options list (distractor count is already verified as sufficient)
        available_distractors = distractors[:required_distractors]  # Take only the required number
        for i in range(num_choices):
            if i == correct_index:
                options.append(correct_conclusion)
            else:
                options.append(available_distractors.pop(0))

        # Verify there are no duplicate options (Fix 4: safety check)
        if len(options) != len(set(options)):
            logger.error(
                f"Duplicate options detected in multiple choice question. "
                f"Options: {options}"
            )
            return None

        question_text = f"Given that {premises}, which of the following conclusions is logically valid?"

        # Apply grammar post-processing to question text and all options
        question_text = self._post_process_mcq_text(question_text)
        options = [self._post_process_mcq_text(opt) for opt in options]
        correct_conclusion = self._post_process_mcq_text(correct_conclusion)

        # Format the option text
        options_text = "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])

        # Compose the full question (including options) - so the LLM can see the options
        question_text_with_options = f"{question_text}\n\n{options_text}"

        # Correct answer letter (A/B/C/D)
        correct_answer_letter = chr(65 + correct_index)

        return {
            'type': QuestionType.MULTIPLE_CHOICE.value,
            'question': question_text_with_options,  # Full question, including options
            'answer': f"The correct answer is {correct_answer_letter}: {correct_conclusion}",
            'correct_answer': correct_answer_letter,  # FIX: return the option letter, not the full text
            'correct_answer_text': correct_conclusion,  # Also provide the full text for reference
            'options': options,
            'reasoning': self._generate_reasoning(body_patterns, head_pattern, entity_labels, relation_labels)
        }

    def _format_premises(self,
                        body_patterns: BodyPatterns,
                        entity_labels: EntityLabels,
                        relation_labels: RelationLabels) -> str:
        """Format the premises.

        Args:
            body_patterns: Either a List[TriplePattern] or a LogicExpression
            entity_labels: Entity label mapping
            relation_labels: Relation label mapping

        Returns:
            The formatted premises string
        """
        # Handle LogicExpression format
        if isinstance(body_patterns, LogicExpression):
            return self._format_logic_expression(body_patterns, entity_labels, relation_labels)

        # Traditional format: List[TriplePattern]
        premises = []
        for s, p, o in body_patterns:
            s_label = entity_labels.get(s, s)
            o_label = entity_labels.get(o, o)
            p_label = relation_labels.get(p, p)
            formatted_statement = self._format_statement(s_label, p_label, o_label, p)
            premises.append(formatted_statement)

        # Improvement: separate multiple premises with commas
        num_premises = len(premises)
        if num_premises == 1:
            return premises[0]
        elif num_premises == 2:
            return f"{premises[0]}, and that {premises[1]}"
        else:
            # For more than two premises, separate with commas and use "and" before the last one
            return ", ".join(premises[:-1]) + f", and that {premises[-1]}"

    def _format_logic_expression(self,
                                 expr: LogicExpression,
                                 entity_labels: EntityLabels,
                                 relation_labels: RelationLabels) -> str:
        """Format a logical expression as natural language.

        Args:
            expr: A LogicExpression object
            entity_labels: Entity label mapping
            relation_labels: Relation label mapping

        Returns:
            The formatted natural language string
        """
        if expr.operator is None:
            # Single fact
            if expr.operands and isinstance(expr.operands[0], Fact):
                fact = expr.operands[0]
                s_label = entity_labels.get(fact.subject, fact.subject)
                o_label = entity_labels.get(fact.object, fact.object)
                p_label = relation_labels.get(fact.predicate, fact.predicate)
                return self._format_statement(s_label, p_label, o_label, fact.predicate)
            return ""

        elif expr.operator == LogicOperator.NOT:
            # NOT expression
            if expr.operands:
                inner = self._format_logic_expression(expr.operands[0], entity_labels, relation_labels)
                return f"it is not the case that {inner}"
            return ""

        elif expr.operator == LogicOperator.AND:
            # AND expression - format as comma-separated list
            formatted_parts = []
            for operand in expr.operands:
                part = self._format_logic_expression(operand, entity_labels, relation_labels)
                formatted_parts.append(part)

            if len(formatted_parts) == 1:
                return formatted_parts[0]
            elif len(formatted_parts) == 2:
                return f"{formatted_parts[0]}, and that {formatted_parts[1]}"
            else:
                return ", ".join(formatted_parts[:-1]) + f", and that {formatted_parts[-1]}"

        elif expr.operator == LogicOperator.OR:
            # OR expression - format with "or"
            formatted_parts = []
            for operand in expr.operands:
                part = self._format_logic_expression(operand, entity_labels, relation_labels)
                formatted_parts.append(part)

            if len(formatted_parts) == 1:
                return formatted_parts[0]
            elif len(formatted_parts) == 2:
                return f"either {formatted_parts[0]}, or {formatted_parts[1]}"
            else:
                # Format: "either A, or B, or C"
                return "either " + ", or ".join(formatted_parts)

        return ""

    def _extract_all_facts(self, expr: LogicExpression) -> List[Fact]:
        """Recursively extract all Facts from a LogicExpression.

        Args:
            expr: LogicExpression object

        Returns:
            List of all Facts in the expression
        """
        facts = []
        if expr.operator is None:
            # Single fact
            if expr.operands and isinstance(expr.operands[0], Fact):
                facts.append(expr.operands[0])
        else:
            # Recursively extract from all operands
            for operand in expr.operands:
                if isinstance(operand, Fact):
                    facts.append(operand)
                elif isinstance(operand, LogicExpression):
                    facts.extend(self._extract_all_facts(operand))
        return facts

    def _count_premises(self, body_patterns: BodyPatterns) -> int:
        """Count the number of premises in body_patterns

        Args:
            body_patterns: Either a List[TriplePattern] or a LogicExpression

        Returns:
            The number of premises
        """
        if isinstance(body_patterns, LogicExpression):
            # For LogicExpression, count all facts recursively
            facts = self._extract_all_facts(body_patterns)
            return len(facts)
        else:
            # For List[TriplePattern], return list length
            return len(body_patterns)

    def _select_question_word_from_statement(self, statement: str, answer_entity_label: str, predicate: str, predicate_label: Optional[str] = None) -> str:
        """Select an appropriate question word based on the semantics of the full statement.

        Strategy:
        1. First check the predicate's semantic features (e.g. location, time, identity)
        2. Then run NER on the full sentence to find the entity type of the answer entity

        Args:
            statement: The full declarative statement
            answer_entity_label: The label of the answer entity
            predicate: The predicate ID
            predicate_label: The text label of the predicate (optional, used for structural analysis)

        Returns:
            An appropriate question word
        """
        # PRIORITY 1: Check predicate semantic category
        # Location predicates should always use "where"
        location_predicates = {
            'P532',  # port of registry
            'P8047', # country of registry
            'P17',   # country
            'P131',  # located in
            'P276',  # location
            'P495',  # country of origin
            'P27',   # country of citizenship
            'P20',   # place of death
            'P19',   # place of birth
            'P159',  # headquarters location
            'P414',  # stock exchange (listed on)
        }

        # Predicates that should ALWAYS use "where" regardless of entity type
        # (e.g., stock exchange is an ORG but we still ask "where is X listed?")
        always_where_predicates = {
            'P414',  # stock exchange - always use "where" for listing location
        }

        # Time predicates should use "when"
        time_predicates = {
            'P571',  # inception
            'P576',  # dissolved
            'P580',  # start time
            'P582',  # end time
            'P585',  # point in time
        }

        # Person predicates should use "who" - teacher/student relations, etc.
        person_predicates = {
            'P1066',  # student of
            'P802',   # student (inverse relation)
            'P185',   # doctoral student
            'P184',   # doctoral advisor
        }

        # Object/Entity predicates should ALWAYS use "what" regardless of NER results
        # These predicates ask "what entity/object" rather than "who" or "where"
        # IMPORTANT: For "adjacent to", we ask "what is adjacent to X?" not "who" or "where"
        # because we're asking for the identity of the adjacent entity, not its location
        always_what_predicates = {
            'P3032',  # adjacent to - asks for adjacent building/entity
            'P2674',  # next crossing downstream - asks for bridge/crossing (not "where")
            'P2673',  # next crossing upstream - asks for bridge/crossing (not "where")
            'P3730',  # next higher rank - asks for rank/position/title (not "who")
            'P3729',  # next lower rank - asks for rank/position/title (not "who")
        }

        # Check for predicates that always use a specific question word
        if predicate in always_where_predicates:
            return 'where'

        if predicate in always_what_predicates:
            return 'what'

        # Use predicate semantics only when the answer entity is location- or time-related
        if self.nlp:
            doc = self.nlp(statement)
            answer_entity_type = self._find_answer_entity_type(doc, answer_entity_label)

            # Use predicate priority only when the answer entity type matches the predicate semantics
            if predicate in location_predicates and answer_entity_type in ['GPE', 'LOC', 'FAC']:
                return 'where'
            elif predicate in time_predicates and answer_entity_type in ['DATE', 'TIME']:
                return 'when'
            elif predicate in person_predicates:
                # For teacher/student relations, etc., force "who"
                return 'who'

        # PRIORITY 2: Analyze the full statement with NER
        if self.nlp:
            return self._select_question_word_from_full_statement(statement, answer_entity_label)
        else:
            raise NotImplementedError("Statement structure analysis not implemented yet.")

    def _find_head_noun(self, tokens) -> Optional[Any]:
        """Find the head noun in a token sequence.

        The head noun is usually:
        1. The last NOUN or PROPN
        2. Or the ROOT of the dependency tree
        """
        # Strategy 1: find the rightmost noun (in English modifiers are on the
        # left and the head word is on the right)
        for token in reversed(tokens):
            if token.pos_ in ['NOUN', 'PROPN']:
                return token

        # Strategy 2: if there are no nouns, return the last token
        return tokens[-1] if tokens else None

    def _map_entity_type_to_question_word(self, entity_type: str) -> str:
        """Map a spaCy entity type to a question word.

        SpaCy recognizes the following built-in entity types:
        PERSON - People, including fictional.
        NORP - Nationalities or religious or political groups.
        FAC - Buildings, airports, highways, bridges, etc.
        ORG - Companies, agencies, institutions, etc.
        GPE - Countries, cities, states.
        LOC - Non-GPE locations, mountain ranges, bodies of water.
        PRODUCT - Objects, vehicles, foods, etc. (Not services.)
        EVENT - Named hurricanes, battles, wars, sports events, etc.
        WORK_OF_ART - Titles of books, songs, etc.
        LAW - Named documents made into laws.
        LANGUAGE - Any named language.
        DATE - Absolute or relative dates or periods.
        TIME - Times smaller than a day.
        PERCENT - Percentage, including "%".
        MONEY - Monetary values, including unit.
        QUANTITY - Measurements, as of weight or distance.
        ORDINAL - "first", "second", etc.
        CARDINAL - Numerals that do not fall under another type.
        """
        mapping = {
            # People and organizations
            'PERSON': 'who',
            'NORP': 'who',  # Nationalities, religious or political groups
            'ORG': 'what',  # Organizations

            # Locations
            'GPE': 'where',  # Countries, cities, states
            'LOC': 'where',  # Non-GPE locations
            'FAC': 'where',  # Buildings, airports, highways, etc.

            # Time
            'DATE': 'when',
            'TIME': 'when',

            # Objects and concepts
            'PRODUCT': 'what',
            'EVENT': 'what',
            'WORK_OF_ART': 'what',
            'LAW': 'what',
            'LANGUAGE': 'what',

            # Numerical values
            'PERCENT': 'how much',
            'MONEY': 'how much',
            'QUANTITY': 'how much',
            'ORDINAL': 'which',
            'CARDINAL': 'what',  # FIX: Changed from 'how many' to 'what'
                                  # Rationale: Most CARDINAL contexts are IDENTITY/VALUE, not COUNT
                                  # COUNT contexts are explicitly detected in _adjust_numeric_question_word()
        }
        return mapping.get(entity_type, 'what')

    def _map_pos_to_question_word(self, pos: str) -> str:
        """Map a part of speech to a question word.

        Used as a fallback when there is no NER information.
        Proper nouns may be person names, place names, etc., so more
        intelligent judgment is needed.
        """
        if pos == 'NOUN':
            return 'what'
        elif pos == 'PROPN':
            # Proper nouns may be person names, place names, organization names, etc.
            # Here we conservatively return 'what', but in practice more complex
            # logic may be required.
            return 'what'
        elif pos == 'NUM':
            return 'when'  # A number may be a year or a quantity
        else:
            return 'what'

    def _select_question_word_from_full_statement(self, statement: str, answer_entity_label: str) -> str:
        """Select an appropriate question word based on NER analysis of the full sentence.

        Strategy:
        1. **PRE-NER pattern check** (new): check obvious patterns first to avoid
           relying on unreliable NER.
        2. Run NER on the full sentence.
        3. Find the entity type corresponding to the answer entity.
        4. If NER cannot recognize it, fall back to POS analysis.
        5. Select an appropriate question word based on the entity type.

        Args:
            statement: The full declarative statement
            answer_entity_label: The label of the answer entity

        Returns:
            An appropriate question word
        """
        # PRIORITY 0: Pre-NER pattern checks
        # These patterns should ALWAYS override NER results because spaCy frequently
        # misclassifies them (e.g., "list of Mexicans" as NORP → "who" instead of "what")
        answer_entity_lower = answer_entity_label.lower()

        # Check for Wikidata special entities (Category:, Template:)
        # These should always use "what" because they are meta-entities, not people/places
        if answer_entity_label.startswith('Category:') or answer_entity_label.startswith('Template:'):
            return 'what'

        # Check for "list of X" pattern
        # "list of X" should always use "what", regardless of what X is
        # spaCy incorrectly identifies "list of Mexicans" as NORP → "who"
        if answer_entity_lower.startswith('list of '):
            return 'what'

        # Check for "universe" suffix (fictional universes)
        # "The King of Fighters universe" should be "what", not "who"
        if 'universe' in answer_entity_lower:
            return 'what'

        # Check for common meta-entities that should use "what"
        # These are categories, types, or classifications, not specific entities
        meta_entity_keywords = ['category', 'template', 'class', 'type', 'kind', 'group']
        for keyword in meta_entity_keywords:
            if answer_entity_lower.startswith(keyword + ':') or answer_entity_lower.startswith('the ' + keyword + ':'):
                return 'what'

        # FIX: Check for biological taxonomy patterns
        # Biological entities (taxonomic ranks) should ALWAYS use "what", not "who" or "where"
        # spaCy frequently misclassifies these as PERSON, GPE, or ORG
        # Examples:
        #   - "Heliconiini (tribe of insects)" -> spaCy says PERSON -> "who" ❌
        #   - "Ischalia (genus of insects)" -> spaCy says GPE -> "where" ❌
        #   - "Trachypithecus (genus of mammals)" -> spaCy says PERSON -> "who" ❌
        # All should use "what" ✓
        biological_category_keywords = [
            # Taxonomic ranks
            'kingdom', 'phylum', 'class', 'order', 'family', 'subfamily', 'genus', 'subgenus',
            'species', 'subspecies', 'tribe', 'subtribe', 'variety', 'form',
            # Biological groups
            'insects', 'mammals', 'birds', 'reptiles', 'amphibians', 'fish', 'plants',
            'bacteria', 'fungi', 'animals', 'organisms', 'invertebrates', 'vertebrates',
            # Specific biological terms
            'taxon', 'taxa', 'clade', 'organism', 'microorganism', 'species group',
        ]

        # Check if answer entity label contains biological keywords
        # Pattern: "Name (taxonomic_rank of biological_group)"
        # Examples: "X (tribe of insects)", "Y (genus of mammals)"
        # Also match: "X (subgenus of X)" where the taxonomic rank itself indicates biology
        for keyword in biological_category_keywords:
            # Match patterns like "(tribe of insects)", "(genus of mammals)", etc.
            if f' of {keyword})' in answer_entity_lower or f' {keyword})' in answer_entity_lower:
                return 'what'

        # Additional pattern: Check for standalone taxonomic ranks even without biological group
        # Examples: "X (subgenus of Y)", "X (genus)", "X (family)"
        # These are still biological entities even if they don't explicitly say "of insects" or "of mammals"
        taxonomic_ranks = [
            'kingdom', 'phylum', 'class', 'order', 'family', 'subfamily', 'genus', 'subgenus',
            'species', 'subspecies', 'tribe', 'subtribe', 'variety', 'form', 'taxon'
        ]
        for rank in taxonomic_ranks:
            # Match patterns like "(subgenus of X)", "(genus)", "(tribe)"
            if f'({rank} of ' in answer_entity_lower or f'({rank})' == answer_entity_lower[-len(rank)-2:]:
                return 'what'

        # FIX for kg_rule_110: Check for software version patterns
        # Pattern: "Software Name Version X" or "Software Name X.Y.Z" or "Software Name vX"
        # These should always use "what", not "how many" (which CARDINAL would give)
        # Examples: "Lightwright Version 3", "Adobe Reader 7.0", "Python v3.9"
        import re
        version_patterns = [
            r'\bversion\s+\d+',  # "Version 3", "version 2"
            r'\bv\d+',  # "v3", "v2"
            r'\b\d+\.\d+',  # "7.0", "3.14"
            r'\breader\s+\d+',  # "Reader 6", "reader 7"
            r'\b(adobe|lightwright|python|java|microsoft)\b.*\d+',  # Software names with numbers
        ]
        for pattern in version_patterns:
            if re.search(pattern, answer_entity_lower):
                return 'what'

        if not self.nlp:
            # If spaCy is unavailable, use a simple fallback
            return self._simple_keyword_check(answer_entity_label)

        try:
            # Run NER on the full sentence
            doc = self.nlp(statement)

            # Find the entity type corresponding to the answer entity in the NER results
            answer_entity_type = self._find_answer_entity_type(doc, answer_entity_label)

            if answer_entity_type:
                # For numeric types, adjust the question word based on context
                if answer_entity_type in ['MONEY', 'QUANTITY', 'CARDINAL']:
                    contextual_numeric = self._adjust_numeric_question_word(statement, answer_entity_label, answer_entity_type)
                    if contextual_numeric:
                        return contextual_numeric
                return self._map_entity_type_to_question_word(answer_entity_type)

            # If NER cannot recognize it, fall back to POS analysis
            answer_pos_type = self._find_answer_pos_type(doc, answer_entity_label)
            if answer_pos_type:
                # For proper nouns, try a context-based keyword check
                if answer_pos_type == 'PROPN':
                    contextual_result = self._contextual_keyword_check(statement, answer_entity_label)
                    if contextual_result:
                        return contextual_result
                return self._map_pos_to_question_word(answer_pos_type)

            # If POS analysis also fails, use simple keyword check as a fallback
            return self._simple_keyword_check(answer_entity_label)

        except (AttributeError, ValueError, RuntimeError) as e:
            # If NER fails, use keyword check
            return self._simple_keyword_check(answer_entity_label)

    def _find_answer_entity_type(self, doc, answer_entity_label: str) -> Optional[str]:
        """Find the entity type corresponding to the answer entity in the NER results.

        Strategy:
        1. First try to exactly match the answer entity label
        2. If exact match fails, try partial match
        3. If partial match fails, return None

        Args:
            doc: The spaCy-processed document object
            answer_entity_label: The label of the answer entity

        Returns:
            The entity type string, or None if not found
        """
        # Exact match: find an entity that exactly matches the answer entity label
        for ent in doc.ents:
            if ent.text == answer_entity_label:
                return ent.label_

        # Partial match: find an entity that contains the answer entity label
        for ent in doc.ents:
            if answer_entity_label in ent.text:
                return ent.label_

        # Reverse partial match: find an entity whose text is contained in the answer entity label
        for ent in doc.ents:
            if ent.text in answer_entity_label:
                return ent.label_

        # If no matching entity is found, return None
        return None

    def _find_answer_pos_type(self, doc, answer_entity_label: str) -> Optional[str]:
        """Find the part-of-speech type corresponding to the answer entity.

        Strategy:
        1. Find a token that matches the answer entity label
        2. Return that token's POS

        Args:
            doc: The spaCy-processed document object
            answer_entity_label: The label of the answer entity

        Returns:
            The POS string, or None if not found
        """
        # Find a token that matches the answer entity label
        for token in doc:
            if token.text == answer_entity_label:
                return token.pos_

        # Partial match: find a token that contains the answer entity label
        for token in doc:
            if answer_entity_label in token.text:
                return token.pos_

        # Reverse partial match: find a token whose text is contained in the answer entity label
        for token in doc:
            if token.text in answer_entity_label:
                return token.pos_

        # If no matching token is found, return None
        return None

    def _contextual_keyword_check(self, statement: str, answer_entity_label: str) -> Optional[str]:
        """Context-based keyword check.

        For proper nouns, check whether the sentence contains keywords indicating
        the entity type.

        Args:
            statement: The full declarative statement
            answer_entity_label: The label of the answer entity

        Returns:
            An appropriate question word, or None if it cannot be determined
        """
        statement_lower = statement.lower()

        # Location-related keywords
        location_keywords = [
            'in', 'at', 'located', 'situated', 'found', 'based', 'headquartered',
            'from', 'to', 'between', 'among', 'around', 'near'
        ]

        # Time-related keywords
        time_keywords = [
            'in', 'on', 'at', 'during', 'before', 'after', 'since', 'until',
            'from', 'to', 'between', 'started', 'ended', 'began', 'finished'
        ]

        # Check whether it is a location
        if any(keyword in statement_lower for keyword in location_keywords):
            # Further confirmation: check whether the answer entity appears near a location keyword
            words = statement.split()
            answer_index = -1
            for i, word in enumerate(words):
                if answer_entity_label in word:
                    answer_index = i
                    break

            if answer_index >= 0:
                # Check whether there is a location keyword before/after the answer entity
                context_window = 3
                start = max(0, answer_index - context_window)
                end = min(len(words), answer_index + context_window + 1)
                context_words = words[start:end]

                if any(keyword in ' '.join(context_words).lower() for keyword in location_keywords):
                    return 'where'

        # Check whether it is a time
        if any(keyword in statement_lower for keyword in time_keywords):
            words = statement.split()
            answer_index = -1
            for i, word in enumerate(words):
                if answer_entity_label in word:
                    answer_index = i
                    break

            if answer_index >= 0:
                context_window = 3
                start = max(0, answer_index - context_window)
                end = min(len(words), answer_index + context_window + 1)
                context_words = words[start:end]

                if any(keyword in ' '.join(context_words).lower() for keyword in time_keywords):
                    return 'when'

        return None

    def _adjust_numeric_question_word(self, statement: str, answer_entity_label: str, entity_type: str) -> Optional[str]:
        """Adjust the question word for numeric types based on context.

        Numbers can appear in three contexts:
        1. COUNT: "how many X are there?" (asking for quantity)
        2. IDENTITY: "what is X?" (asking for entity identity)
        3. VALUE: "what/how much is X?" (asking for numeric value)

        CRITICAL FIX (kg_rule_105):
        - Before: CARDINAL -> 'how many' (always)
        - After:  CARDINAL -> context-dependent ('what' for identity, 'how many' for count)

        Args:
            statement: The full declarative statement
            answer_entity_label: The label of the answer entity
            entity_type: Entity type

        Returns:
            The adjusted question word, or None if no adjustment is needed
        """
        statement_lower = statement.lower()

        # =========================================================================
        # PRIORITY 1: COUNT Context Detection (Most Specific)
        # =========================================================================
        # Pattern: "how many [countable nouns]..."
        # Example: "How many students are there?"
        #
        # NOTE: This must be checked BEFORE identity detection to avoid conflicts
        # Example: "How many prime factors does X have?" should use 'how many', not 'what'
        countable_keywords = [
            'population', 'people', 'persons', 'individuals', 'citizens',
            'students', 'workers', 'employees', 'members', 'participants',
            'items', 'products', 'books', 'cars', 'houses', 'buildings',
            'companies', 'organizations', 'countries', 'cities'
        ]

        if any(keyword in statement_lower for keyword in countable_keywords):
            return 'how many'

        # =========================================================================
        # PRIORITY 2: IDENTITY Context Detection
        # =========================================================================
        # Pattern: "X is [identity_noun] of Y" → "what is X [identity_noun] of?"
        # Example: "401 is a prime factor of X" → "what is 401 a prime factor of?"
        #
        # Root Cause (kg_rule_105):
        #   - System was using 'how many' for all CARDINALs
        #   - But "how many is 401 a prime factor of?" is grammatically wrong
        #   - Should be "what is 401 a prime factor of?"
        #
        # Fix: Detect identity context and use 'what' instead of 'how many'
        identity_indicators = [
            # Mathematical identity patterns
            'factor of', 'divisor of', 'prime factor', 'prime divisor',
            'multiple of', 'power of', 'root of', 'square root',
            'solution to', 'result of', 'value of',

            # Structural/compositional identity
            'part of', 'component of', 'element of', 'member of',
            'piece of', 'section of', 'portion of',

            # Classification identity
            'instance of', 'example of', 'type of', 'kind of',
            'category of', 'class of',

            # Relationship identity
            'equals', 'represents', 'constitutes', 'forms',
            'comprises', 'consists of',
        ]

        # Check if statement contains identity indicators
        for indicator in identity_indicators:
            if indicator in statement_lower:
                # Validated: This is an identity context
                # Use 'what' to ask for entity identity, not 'how many' for count
                #
                # Example transformations:
                #   "401 is a prime factor of X" → "what is 401 a prime factor of?"
                #   "2 is a divisor of X" → "what is 2 a divisor of?"
                #   "5 is part of X" → "what is 5 part of?"
                return 'what'

        # =========================================================================
        # PRIORITY 3: VALUE Context (Default for MONEY and CARDINAL)
        # =========================================================================
        # Pattern: "What is the [measurement]?" or "How much is X?"
        # Example: "What is the temperature?" "How much does it cost?"

        if entity_type == 'MONEY':
            return 'how much'
        elif entity_type == 'CARDINAL':
            # CRITICAL FIX: Changed default from 'how many' to 'what'
            #
            # Rationale:
            #   - Most CARDINAL contexts that aren't COUNT are asking for identity/value
            #   - 'what' is grammatically safer than 'how many' in general
            #   - COUNT contexts are explicitly detected above
            #
            # Impact:
            #   Before: "how many is 401 a prime factor of?" ❌
            #   After:  "what is 401 a prime factor of?" ✅
            return 'what'

        return None

    def _select_question_word(self, entity_id: str, entity_label: str) -> str:
        """Select an appropriate question word based on entity type, using spaCy NER (legacy interface preserved)."""
        if not self.nlp:
            # If spaCy is unavailable, use a simple fallback
            return self._simple_keyword_check(entity_label)

        try:
            # Use spaCy for named entity recognition
            doc = self.nlp(entity_label)

            # Extract entity types
            entity_types = [ent.label_ for ent in doc.ents]

            # If spaCy recognized an entity type, use it
            if entity_types:
                return self._get_ner_suggestion(entity_types, entity_label)

            # If no entity type was recognized, use simple keyword check as a fallback
            return self._simple_keyword_check(entity_label)

        except (AttributeError, ValueError, RuntimeError) as e:
            # If NER fails, use keyword check
            return self._simple_keyword_check(entity_label)

    def _get_ner_suggestion(self, entity_types: List[str], entity_label: str = "") -> str:
        """Get a question word suggestion from NER types.

        When there are multiple NER types, select by priority.
        Priority: PERSON > DATE/TIME > GPE/LOCATION > ORGANIZATION/FACILITY
        """
        # Process NER types by priority.
        # PERSON has the highest priority to avoid misclassifying person names as places.
        if 'PERSON' in entity_types:
            return "who"
        elif 'DATE' in entity_types or 'TIME' in entity_types:
            return "when"
        elif 'GPE' in entity_types:
            return "where"
        elif 'LOC' in entity_types:
            return "where"
        elif 'ORG' in entity_types:
            return "what"
        elif 'FAC' in entity_types:
            return "what"
        else:
            return "what"  # Default

    def _simple_keyword_check(self, entity_label: str) -> str:
        """Simple keyword check - used only when NER cannot recognize the entity."""
        # Year pattern - the only special case, because spaCy cannot reliably
        # recognize pure numbers
        if self._is_year_pattern(entity_label):
            return "when"

        # Default to "what"
        return "what"

    def _is_year_pattern(self, text: str) -> bool:
        """Check whether this is a year pattern."""
        # Check whether it is a 4-digit number (year)
        if text.isdigit() and len(text) == 4:
            year = int(text)
            # Reasonable year range
            return 1000 <= year <= 3000
        return False

    def _smart_lowercase(self, text: str) -> str:
        """Smart lowercase conversion that protects proper nouns and special formats."""
        if not text:
            return text

        import re

        # First normalize Template and Category names
        text = self._normalize_template_category_names(text)

        # Process word by word rather than lowercasing globally
        words = text.split()
        result_words = []

        # Full patterns for special formats that need protection
        special_patterns = [
            r'^Template:[A-Za-z0-9_:-]+$',
            r'^Category:[A-Za-z0-9_:-]+$',
            r'^File:[A-Za-z0-9_:-]+$',
            r'^User:[A-Za-z0-9_:-]+$',
            r'^COVID-19$',
        ]

        # Function words that should be lowercased
        lowercase_words = {
            'and', 'or', 'but', 'in', 'on', 'at', 'by', 'for', 'with', 'without',
            'of', 'to', 'from', 'about', 'over', 'under', 'through', 'during',
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'can', 'could', 'should', 'shall', 'may', 'might', 'must'
        }

        for i, word in enumerate(words):
            # Check whether it is a special format
            is_special_format = any(re.match(pattern, word) for pattern in special_patterns)

            if is_special_format:
                # Keep special formats unchanged
                result_words.append(word)
            elif i == 0:
                # Capitalize the first word of the sentence (unless it is a special marker)
                result_words.append(word.lower().capitalize())
            elif word.lower() in lowercase_words:
                # Lowercase function words
                result_words.append(word.lower())
            elif word.lower() == 'i':
                # "I" is always capitalized
                result_words.append('I')
            else:
                # For other words: if originally capitalized (likely a proper noun), keep as is;
                # if originally lowercase, keep lowercase
                if word and word[0].isupper():
                    result_words.append(word)  # Preserve the original capitalization of proper nouns
                else:
                    result_words.append(word.lower())

        return ' '.join(result_words)

    def _normalize_template_category_names(self, text: str) -> str:
        """Normalize Template and Category name formats."""
        import re

        # Fix "template : xxx" format -> Template:xxx
        text = re.sub(r'\btemplate\s*:\s*([a-zA-Z0-9_:-]+)',
                     lambda m: f"Template:{m.group(1)}", text, flags=re.IGNORECASE)

        # Fix "category : xxx" format -> Category:xxx
        text = re.sub(r'\bcategory\s*:\s*([a-zA-Z0-9_:-]+)',
                     lambda m: f"Category:{m.group(1)}", text, flags=re.IGNORECASE)

        # Ensure COVID-19 keeps the correct capitalization
        text = re.sub(r'\bcovid-19\b', 'COVID-19', text, flags=re.IGNORECASE)

        return text

    def _format_conclusion(self,
                          head_pattern: TriplePattern,
                          entity_labels: EntityLabels,
                          relation_labels: RelationLabels) -> str:
        """Format the conclusion."""
        s, p, o = head_pattern
        s_label = entity_labels.get(s, s)
        o_label = entity_labels.get(o, o)
        p_label = relation_labels.get(p, p)
        return self._format_statement(s_label, p_label, o_label, p)

    def _extract_description_from_label(self, label_with_desc: str) -> tuple:
        """Extract the pure label and the description from a label with description.

        Args:
            label_with_desc: Format may be "Label (description)" or "Label"

        Returns:
            tuple: (pure_label, description)
        """
        if not label_with_desc:
            return "", ""

        # Find the last " (" and ")" to extract the description.
        # Note: the description may contain parentheses, so we look for the last match.
        import re
        match = re.match(r'^(.+?)\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)$', label_with_desc)
        if match:
            pure_label = match.group(1).strip()
            description = match.group(2).strip()
            return pure_label, description

        return label_with_desc, ""

    def _extract_pure_entities_from_statement(self, statement: str) -> str:
        """Remove all parenthetical entity descriptions from the statement,
        keeping only the pure labels.

        This is critical for wh-question generation. When a statement contains entities
        with parenthetical descriptions like "quasi-national park of Japan (National park
        in Japan. It is designated in...)", spaCy misidentifies the subject and verb
        because it treats the parenthetical clause as the main clause.

        Args:
            statement: Statement possibly containing entities with "(description)"

        Returns:
            Statement with all parenthetical descriptions removed

        Example:
            Input: "quasi-national park of Japan (National park in Japan. It is...) has
                    the next higher rank of national park (nature park...)"
            Output: "quasi-national park of Japan has the next higher rank of national park"
        """
        import re
        # Pattern to match " (description)" where description may contain nested parentheses
        # We use a regex that handles nested parentheses up to reasonable depth
        # This pattern matches: space + opening paren + content (may have nested parens) + closing paren

        # Strategy: Iteratively remove the outermost parenthetical groups
        # to handle nested parentheses correctly
        cleaned = statement
        max_iterations = 10  # Safety limit to prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            # Find and remove outermost parenthetical descriptions
            # Pattern: space + ( + content without unmatched parens + )
            # We use a greedy match to capture complete parenthetical groups
            prev_cleaned = cleaned

            # Match pattern: " (content)" where content has balanced parens
            # Use a simpler approach: match " (" then find the matching ")"
            import re
            result = []
            i = 0
            while i < len(cleaned):
                if i + 1 < len(cleaned) and cleaned[i:i+2] == ' (':
                    # Found potential start of parenthetical description
                    # Find the matching closing parenthesis
                    paren_level = 1
                    j = i + 2
                    while j < len(cleaned) and paren_level > 0:
                        if cleaned[j] == '(':
                            paren_level += 1
                        elif cleaned[j] == ')':
                            paren_level -= 1
                        j += 1

                    if paren_level == 0:
                        # Found matching closing paren, skip the entire group
                        i = j
                        continue
                    else:
                        # Unmatched, keep the opening " ("
                        result.append(cleaned[i])
                        i += 1
                else:
                    result.append(cleaned[i])
                    i += 1

            cleaned = ''.join(result)

            # If no change, we're done
            if cleaned == prev_cleaned:
                break

            iteration += 1

        # Clean up extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned

    def _format_statement(self,
                         subject: str,
                         predicate: str,
                         object_: str,
                         property_id: Optional[str] = None) -> str:
        """Format a single statement (subject-predicate-object) into correct English grammar.

        Supports type-aware formatting: extracts description information from the
        subject label to select a semantically correct formatting pattern.
        """

        # Extract description information from the subject for type-aware formatting
        pure_subject, subject_description = self._extract_description_from_label(subject)

        # Use the intelligent grammar analyzer to analyze the full sentence structure
        analysis = self.grammar_analyzer.analyze_sentence_structure(pure_subject, predicate, object_, property_id)

        # Get the analysis results
        suggested_verb = analysis['suggested_verb']
        suggested_preposition = analysis['suggested_preposition']

        # Use the Wikidata property formatter for intelligent formatting.
        # Pass subject_description to enable type-aware formatting.
        if property_id:
            formatted_statement = wikidata_formatter.format_statement(
                subject, predicate, object_, property_id,
                subject_description=subject_description
            )
            return self._apply_context_aware_grammar(formatted_statement, subject, predicate, object_)

        # If there is no property ID, use grammar-aware sentence construction
        return self._build_grammatically_aware_sentence(subject, predicate, object_, suggested_verb, suggested_preposition, analysis)

    def _build_grammatically_aware_sentence(self, subject: str, predicate: str, object_: str,
                                          suggested_verb: str, suggested_preposition: str, analysis: dict) -> str:
        """Build a grammar-aware sentence, avoiding article conflicts."""

        # 1. Analyze the grammatical features of each component
        subject_info = self._analyze_grammatical_features(subject)
        object_info = self._analyze_grammatical_features(object_)
        predicate_info = self._analyze_predicate_features(predicate)

        # 2. Intelligently add articles based on grammatical context
        formatted_subject = self._add_contextual_article(subject, subject_info, 'subject')
        formatted_object = self._add_contextual_article(object_, object_info, 'object', predicate_info)

        # 3. Process the predicate
        predicate_analysis = analysis['predicate_analysis']
        formatted_predicate = self.article_analyzer.add_article_to_predicate(predicate)

        # 4. Intelligently combine the sentence, avoiding repetition
        if suggested_preposition and not predicate_analysis['ends_with_preposition']:
            statement = f"{formatted_subject} {suggested_verb} {formatted_predicate} {suggested_preposition} {formatted_object}"
        else:
            statement = f"{formatted_subject} {suggested_verb} {formatted_predicate} {formatted_object}"

        # 5. Post-processing: check and fix obvious grammatical errors
        return self._post_process_sentence(statement)

    def _analyze_grammatical_features(self, noun_phrase: str) -> dict:
        """Analyze the grammatical features of a noun phrase."""
        return {
            'is_wikidata_entity': ':' in noun_phrase and any(noun_phrase.lower().startswith(prefix)
                                  for prefix in ['category:', 'template:', 'file:', 'user:']),
            'has_determiner': self._has_determiner_simple(noun_phrase),
            'is_proper_noun': self._is_proper_noun_simple(noun_phrase),
            'is_plural': self.article_analyzer.is_subject_plural(noun_phrase)
        }

    def _has_determiner_simple(self, noun_phrase: str) -> bool:
        """Simple determiner check."""
        determiners = {
            'a', 'an', 'the', 'this', 'that', 'these', 'those',
            'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        first_word = noun_phrase.split()[0].lower()
        return first_word in determiners

    def _is_proper_noun_simple(self, noun_phrase: str) -> bool:
        """Simple proper-noun check."""
        return noun_phrase[0].isupper() if noun_phrase else False

    def _analyze_predicate_features(self, predicate: str) -> dict:
        """Analyze the grammatical features of a predicate."""
        return {
            'contains_category': 'category' in predicate.lower(),
            'contains_template': 'template' in predicate.lower(),
            'contains_population': 'populat' in predicate.lower()
        }

    def _add_contextual_article(self, noun_phrase: str, info: dict, role: str, predicate_info: dict = None) -> str:
        """Intelligently add an article based on context."""
        # If a determiner is already present, return as is
        if info['has_determiner']:
            return noun_phrase

        # Handling of special Wikidata entities
        if info['is_wikidata_entity']:
            # For Category: and Template:, use the definite article in specific contexts
            if predicate_info and (predicate_info['contains_category'] or predicate_info['contains_population']):
                return f"the {noun_phrase}"
            else:
                return noun_phrase

        # In other cases, use the standard logic of ArticleAnalyzer
        return self.article_analyzer.add_article(noun_phrase)

    def _apply_context_aware_grammar(self, statement: str, subject: str, predicate: str, object_: str) -> str:
        """Apply context-aware grammatical corrections to a preformatted sentence."""
        # Check for obvious grammatical error patterns
        return self._post_process_sentence(statement)

    def _post_process_sentence(self, statement: str) -> str:
        """Post-process the sentence to fix obvious grammatical errors."""
        # Remove extra spaces
        import re
        statement = re.sub(r'\s+', ' ', statement).strip()

        # Check and fix double-article issues (but do not use global replacement).
        # For example: "the category the Category:" -> "the Category:"
        # Or: "the category Category:" -> "the Category:"
        words = statement.split()
        processed_words = []

        i = 0
        while i < len(words):
            word = words[i]

            # Pattern 1: "the category a/the Category:" or "the category a/the Template:"
            # Fix: "the category a/the Category:" -> "the Category:"
            if (word.lower() == 'the' and i + 2 < len(words) and
                words[i + 1].lower() == 'category' and
                words[i + 2].lower() in ['a', 'the'] and
                i + 3 < len(words) and
                (':' in words[i + 3])):

                # Skip "the category a/the", keep only "the Category:XXX"
                processed_words.append('the')
                processed_words.append(words[i + 3])  # Category:XXX
                i += 4

            # Pattern 2: "the category Category:" or "the category Template:" (NEW FIX)
            # Fix: "the category Category:XXX" -> "the Category:XXX"
            elif (word.lower() == 'the' and i + 1 < len(words) and
                  words[i + 1].lower() == 'category' and
                  i + 2 < len(words) and
                  ':' in words[i + 2] and
                  any(words[i + 2].startswith(prefix + ':') for prefix in ['Category', 'Template', 'category', 'template'])):

                # Skip "the category", keep only "the Category:XXX"
                processed_words.append('the')
                processed_words.append(words[i + 2])  # Category:XXX or Template:XXX
                i += 3

            # Pattern 3: "the template Template:" (similar to pattern 2)
            # Fix: "the template Template:XXX" -> "the Template:XXX"
            elif (word.lower() == 'the' and i + 1 < len(words) and
                  words[i + 1].lower() == 'template' and
                  i + 2 < len(words) and
                  ':' in words[i + 2] and
                  any(words[i + 2].startswith(prefix + ':') for prefix in ['Category', 'Template', 'category', 'template'])):

                # Skip "the template", keep only "the Template:XXX"
                processed_words.append('the')
                processed_words.append(words[i + 2])  # Category:XXX or Template:XXX
                i += 3

            else:
                processed_words.append(word)
                i += 1

        return ' '.join(processed_words)

    def _post_process_mcq_text(self, text: str) -> str:
        """Post-process multiple choice question text for grammar issues

        Fixes:
        1. Ensure questions end with '?'
        2. Normalize spacing around parentheses
        3. Remove multiple consecutive spaces
        4. Fix capitalization issues
        """
        import re

        # Remove multiple consecutive spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Normalize spacing around parentheses: remove space after ( and before )
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)

        # Ensure question sentences end with '?'
        # Only apply to main question text (not options)
        if 'which of the following' in text.lower() and not text.rstrip().endswith('?'):
            text = text.rstrip('.!') + '?'

        return text

    def _generate_reasoning(self,
                           body_patterns: BodyPatterns,
                           head_pattern: TriplePattern,
                           entity_labels: EntityLabels,
                           relation_labels: RelationLabels) -> str:
        """Generate the reasoning process."""
        premises = self._format_premises(body_patterns, entity_labels, relation_labels)
        conclusion = self._format_conclusion(head_pattern, entity_labels, relation_labels)
        return f"From the premises '{premises}', we can logically infer '{conclusion}'."

    def _clean_variable_name(self, var_name: str) -> str:
        """Clean a variable name by removing extraneous characters."""
        # Remove a leading parenthesis
        if var_name.startswith('('):
            var_name = var_name[1:]
        # Remove trailing parentheses, commas, etc.
        var_name = var_name.rstrip('),')
        return var_name

    def _get_entity_labels(self,
                          instantiation: Dict[str, str],
                          patterns: Optional[List[TriplePattern]] = None) -> EntityLabels:
        """Get entity labels, including description information."""
        labels = {}

        # First process the entities in the instantiation mapping
        for var, entity_id in instantiation.items():
            clean_var = self._clean_variable_name(var)
            if entity_id not in self.entity_cache:
                try:
                    # Use the knowledge graph's format_entity method to show the description
                    if hasattr(self.kg, 'format_entity'):
                        formatted_entity = self.kg.format_entity(entity_id, show_description=True)
                        self.entity_cache[entity_id] = formatted_entity
                    else:
                        # If the KG does not support format_entity, fall back to showing only the label
                        label = self.kg.get_entity_label(entity_id) or entity_id
                        description = self.kg.get_entity_description(entity_id) if hasattr(self.kg, 'get_entity_description') else None

                        # If there is a description, append it to the label
                        if description and description != "<<NO_DESCRIPTION>>":
                            self.entity_cache[entity_id] = f"{label} ({description})"
                        else:
                            self.entity_cache[entity_id] = label
                except (AttributeError, TypeError, KeyError):
                    self.entity_cache[entity_id] = entity_id
            labels[var] = self.entity_cache[entity_id]
            labels[clean_var] = self.entity_cache[entity_id]  # Add the cleaned variable name
            labels[entity_id] = self.entity_cache[entity_id]

        # If there are additional patterns, process all entities within them
        if patterns:
            for pattern in patterns:
                for entity in pattern:
                    clean_entity = self._clean_variable_name(entity)
                    # Check whether it is a variable (present in instantiation)
                    if clean_entity in instantiation:
                        actual_entity = instantiation[clean_entity]
                        if actual_entity not in self.entity_cache:
                            try:
                                # Use the knowledge graph's format_entity method to show the description
                                if hasattr(self.kg, 'format_entity'):
                                    formatted_entity = self.kg.format_entity(actual_entity, show_description=True)
                                    self.entity_cache[actual_entity] = formatted_entity
                                else:
                                    # If the KG does not support format_entity, fall back to showing only the label
                                    label = self.kg.get_entity_label(actual_entity) or actual_entity
                                    description = self.kg.get_entity_description(actual_entity) if hasattr(self.kg, 'get_entity_description') else None

                                    # If there is a description, append it to the label
                                    if description and description != "<<NO_DESCRIPTION>>":
                                        self.entity_cache[actual_entity] = f"{label} ({description})"
                                    else:
                                        self.entity_cache[actual_entity] = label
                            except (AttributeError, TypeError, KeyError):
                                self.entity_cache[actual_entity] = actual_entity
                        labels[entity] = self.entity_cache[actual_entity]
                        labels[clean_entity] = self.entity_cache[actual_entity]
                    elif entity.startswith('Q') and entity not in labels:
                        # Direct entity ID
                        if entity not in self.entity_cache:
                            try:
                                # Use the knowledge graph's format_entity method to show the description
                                if hasattr(self.kg, 'format_entity'):
                                    formatted_entity = self.kg.format_entity(entity, show_description=True)
                                    self.entity_cache[entity] = formatted_entity
                                else:
                                    # If the KG does not support format_entity, fall back to showing only the label
                                    label = self.kg.get_entity_label(entity) or entity
                                    description = self.kg.get_entity_description(entity) if hasattr(self.kg, 'get_entity_description') else None

                                    # If there is a description, append it to the label
                                    if description and description != "<<NO_DESCRIPTION>>":
                                        self.entity_cache[entity] = f"{label} ({description})"
                                    else:
                                        self.entity_cache[entity] = label
                            except (AttributeError, TypeError, KeyError):
                                self.entity_cache[entity] = entity
                        labels[entity] = self.entity_cache[entity]

        return labels

    def _get_relation_labels(self, patterns: List[TriplePattern]) -> RelationLabels:
        """Get relation labels."""
        labels = {}
        for s, p, o in patterns:
            if p.startswith('P'):
                # First try to retrieve from the cache
                if p not in self.property_cache:
                    try:
                        # Use the knowledge graph interface to get the property label
                        label = self.kg.get_property_label(p) or p
                        self.property_cache[p] = label
                    except (AttributeError, TypeError, KeyError):
                        # On failure, try using the PropertyManager
                        try:
                            label = self.grammar_analyzer.property_manager.get_property_label(p)
                            self.property_cache[p] = label
                        except (AttributeError, TypeError, KeyError):
                            self.property_cache[p] = p
                labels[p] = self.property_cache[p]
        return labels

    def _get_functionality(self, predicate: str) -> float:
        """Get the functionality of a relation."""
        try:
            # Check whether the knowledge graph supports functionality queries
            if hasattr(self.kg, 'functionality'):
                return self.kg.functionality(predicate)
            else:
                # If not supported, return the default value 0.5
                return 0.5
        except (AttributeError, TypeError, ValueError):
            # On error, return the default value
            return 0.5

    def _get_inverse_functionality(self, predicate: str) -> float:
        """Get the inverse functionality of a relation."""
        try:
            # Check whether the knowledge graph supports inverse functionality queries
            if hasattr(self.kg, 'inverse_functionality'):
                return self.kg.inverse_functionality(predicate)
            else:
                # If not supported, return the default value 0.5
                return 0.5
        except (AttributeError, TypeError, ValueError):
            # On error, return the default value
            return 0.5

    def _get_entity_label_from_kg(self, entity_id: str) -> str:
        """Get an entity label from the KG."""
        if entity_id in self.entity_cache:
            return self.entity_cache[entity_id]

        try:
            if hasattr(self.kg, 'format_entity'):
                label = self.kg.format_entity(entity_id, show_description=True)
            else:
                label = self.kg.get_entity_label(entity_id) or entity_id
            self.entity_cache[entity_id] = label
            return label
        except (AttributeError, TypeError, KeyError):
            return entity_id

    def _get_relation_label_from_kg(self, relation_id: str) -> str:
        """Get a relation label from the KG."""
        if relation_id in self.property_cache:
            return self.property_cache[relation_id]

        try:
            label = self.kg.get_property_label(relation_id) or relation_id
            self.property_cache[relation_id] = label
            return label
        except (AttributeError, TypeError, KeyError):
            return relation_id
