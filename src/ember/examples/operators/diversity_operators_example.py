import os 
import logging

# Set global logging level to ERROR
logging.basicConfig(level=logging.ERROR)

os.environ["EMBER_LOGGING_LEVEL"] = "ERROR"

# from ember.core.registry.model.model_module.lm import LMModule, LMModuleConfig
from ember.core.registry.model.config.settings import initialize_registry
from ember.core.registry.model.base.services.model_service import ModelService
from ember.core.utils.eval.evaluators import (DiversityCosineSimilarityEvaluator, 
                                              DiversityEnsembledEvaluator, 
                                              DiversityEditDistanceEvaluator,
                                              DiversityNoveltyEvaluator,
                                              DiversityCompressionEvaluator
)

from ember.core.registry.model.providers.openai.openai_provider import create_openai_embedding_model


model_registry = initialize_registry()
logging.info(model_registry.list_models())

text_embedding_ada_002 = create_openai_embedding_model("text-embedding-ada-002")

# List of text that represents completely n
very_diverse_text = ["Bananas don't belong in briefcases, but socks and t-shirts do!", 
                     "Abraham Lincoln was the 16th president of the United States of America", 
                     "ERROR 404: Index Not Found"]

# This group of text all rephrase the same request, except 
different_words_not_diverse_strs = ["Could you please lend me a hand with this?", 
                                    "Might you assist me with a task?", 
                                    "Can you spare a second to help me do something?"]

repetition_strs = ["This is a sample text with lots of repetition.", 
                   "This is a sample text with lots of repetition.",
                   "This is a sample text with lots of repetition."]

# List of sample strings that have varying levels of diversity:
test_strings = [very_diverse_text, different_words_not_diverse_strs, repetition_strs]


# Measure Cosine similarity
cosine_similarity_evaluator = DiversityCosineSimilarityEvaluator(text_embedding_ada_002)

print("\n" + "=" * 50 )
print("Cosine Similarity Evaluator\n")
for i in range(len(test_strings)):
    print(f"Computing cosine-similarity for the following strings: ")
    for j in range(len(test_strings[i])):
        print(f"String {j+1}: {test_strings[i][j]}")
    score: float = cosine_similarity_evaluator.evaluate(system_output=test_strings[i]).score
    print(f"Diversity score: {score}\n")


# Measure Edit Distance
print("=" * 50 + "\nEdit Distance Evaluator\n")
edit_distance_evaluator = DiversityEditDistanceEvaluator()

for i in range(len(test_strings)):
    print(f"Computing Edit-Distance for the following strings: ")
    for j in range(len(test_strings[i])):
        print(f"String {j+1}: {test_strings[i][j]}")
    score: float = edit_distance_evaluator.evaluate(system_output=test_strings[i]).score
    print(f"Edit-Distance score: {score}\n")
print("=" * 50 + "\n")


# Measure Novelty
print("=" * 50 + "\nNovelty Evaluator\n")
novelty_evaluator = DiversityNoveltyEvaluator()

for i in range(len(test_strings)):
    print(f"Computing Novelty for the following strings: ")
    for j in range(len(test_strings[i])):
        print(f"String {j+1}: {test_strings[i][j]}")
    score: float = novelty_evaluator.evaluate(system_output=test_strings[i]).score
    print(f"Novelty score: {score}\n")
print("=" * 50 + "\n")


# Measure Compression Ratio
print("=" * 50 + "\nCompression Ratio Evaluator\n")
novelty_evaluator = DiversityCompressionEvaluator()

for i in range(len(test_strings)):
    print(f"Computing Compression Ratio for the following strings: ")
    for j in range(len(test_strings[i])):
        print(f"String {j+1}: {test_strings[i][j]}")
    score: float = novelty_evaluator.evaluate(system_output=test_strings[i]).score
    print(f"Compression Ratio: {score}\n")
print("=" * 50 + "\n")


# Measure Ensembled Diversity
print("=" * 50 + "\nEnsembled Diversity Evaluator\n")
novelty_evaluator = DiversityCompressionEvaluator()

for i in range(len(test_strings)):
    print(f"Computing Ensembled Diversity Score for the following strings: ")
    for j in range(len(test_strings[i])):
        print(f"String {j+1}: {test_strings[i][j]}")
    score: float = novelty_evaluator.evaluate(system_output=test_strings[i]).score
    print(f"Ensembled Diversity Score: {score}\n")
print("=" * 50 + "\n")