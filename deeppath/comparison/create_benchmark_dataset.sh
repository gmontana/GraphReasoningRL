#!/bin/bash
# Script to create a simplified but realistic benchmark dataset for comparing implementations

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the repository root directory (one level up from script)
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "=== Creating simplified benchmark dataset ==="

# Relation to benchmark
relation="athletePlaysForTeam"

# Create directories
mkdir -p "$REPO_ROOT/NELL-995/tasks/$relation"

# Create a structured knowledge graph with clear paths between entities
# This will ensure the path finding algorithm can find valid paths
cat > "$REPO_ROOT/NELL-995/tasks/$relation/graph.txt" << 'EOL'
concept_athlete_michael_jordan concept_sportsTeam_chicago_bulls concept:athletePlaysForTeam
concept_sportsTeam_chicago_bulls concept_athlete_michael_jordan concept:athletePlaysForTeam_inv
concept_athlete_stephen_curry concept_sportsTeam_golden_state_warriors concept:athletePlaysForTeam
concept_sportsTeam_golden_state_warriors concept_athlete_stephen_curry concept:athletePlaysForTeam_inv
concept_athlete_lebron_james concept_sportsTeam_cleveland_cavaliers concept:athletePlaysForTeam
concept_sportsTeam_cleveland_cavaliers concept_athlete_lebron_james concept:athletePlaysForTeam_inv
concept_athlete_michael_jordan concept_sport_basketball concept:athletePlaysSport
concept_sport_basketball concept_athlete_michael_jordan concept:athletePlaysSport_inv
concept_athlete_stephen_curry concept_sport_basketball concept:athletePlaysSport
concept_sport_basketball concept_athlete_stephen_curry concept:athletePlaysSport_inv
concept_athlete_lebron_james concept_sport_basketball concept:athletePlaysSport
concept_sport_basketball concept_athlete_lebron_james concept:athletePlaysSport_inv
concept_sportsTeam_chicago_bulls concept_sport_basketball concept:teamPlaysSport
concept_sport_basketball concept_sportsTeam_chicago_bulls concept:teamPlaysSport_inv
concept_sportsTeam_golden_state_warriors concept_sport_basketball concept:teamPlaysSport
concept_sport_basketball concept_sportsTeam_golden_state_warriors concept:teamPlaysSport_inv
concept_sportsTeam_cleveland_cavaliers concept_sport_basketball concept:teamPlaysSport
concept_sport_basketball concept_sportsTeam_cleveland_cavaliers concept:teamPlaysSport_inv
concept_athlete_michael_jordan concept_sportsLeague_nba concept:athletePlaysInLeague
concept_sportsLeague_nba concept_athlete_michael_jordan concept:athletePlaysInLeague_inv
concept_athlete_stephen_curry concept_sportsLeague_nba concept:athletePlaysInLeague
concept_sportsLeague_nba concept_athlete_stephen_curry concept:athletePlaysInLeague_inv
concept_athlete_lebron_james concept_sportsLeague_nba concept:athletePlaysInLeague
concept_sportsLeague_nba concept_athlete_lebron_james concept:athletePlaysInLeague_inv
concept_sportsTeam_chicago_bulls concept_sportsLeague_nba concept:teamPlaysInLeague
concept_sportsLeague_nba concept_sportsTeam_chicago_bulls concept:teamPlaysInLeague_inv
concept_sportsTeam_golden_state_warriors concept_sportsLeague_nba concept:teamPlaysInLeague
concept_sportsLeague_nba concept_sportsTeam_golden_state_warriors concept:teamPlaysInLeague_inv
concept_sportsTeam_cleveland_cavaliers concept_sportsLeague_nba concept:teamPlaysInLeague
concept_sportsLeague_nba concept_sportsTeam_cleveland_cavaliers concept:teamPlaysInLeague_inv
EOL

# Create training data with a few samples
cat > "$REPO_ROOT/NELL-995/tasks/$relation/train_pos" << 'EOL'
concept_athlete_michael_jordan concept_sportsTeam_chicago_bulls concept:athletePlaysForTeam+
concept_athlete_stephen_curry concept_sportsTeam_golden_state_warriors concept:athletePlaysForTeam+
concept_athlete_lebron_james concept_sportsTeam_cleveland_cavaliers concept:athletePlaysForTeam+
EOL

# Create testing data
cp "$REPO_ROOT/NELL-995/tasks/$relation/train_pos" "$REPO_ROOT/NELL-995/tasks/$relation/test_pos"

# Create train.pairs and sort_test.pairs
cp "$REPO_ROOT/NELL-995/tasks/$relation/train_pos" "$REPO_ROOT/NELL-995/tasks/$relation/train.pairs"
cp "$REPO_ROOT/NELL-995/tasks/$relation/test_pos" "$REPO_ROOT/NELL-995/tasks/$relation/sort_test.pairs"

# Create entity2id.txt
cat > "$REPO_ROOT/NELL-995/entity2id.txt" << 'EOL'
concept_athlete_michael_jordan 0
concept_sportsTeam_chicago_bulls 1
concept_athlete_stephen_curry 2
concept_sportsTeam_golden_state_warriors 3
concept_athlete_lebron_james 4
concept_sportsTeam_cleveland_cavaliers 5
concept_sport_basketball 6
concept_sportsLeague_nba 7
EOL

# Create relation2id.txt
cat > "$REPO_ROOT/NELL-995/relation2id.txt" << 'EOL'
concept:athletePlaysForTeam 0
concept:athletePlaysForTeam_inv 1
concept:athletePlaysSport 2
concept:athletePlaysSport_inv 3
concept:teamPlaysSport 4
concept:teamPlaysSport_inv 5
concept:athletePlaysInLeague 6
concept:athletePlaysInLeague_inv 7
concept:teamPlaysInLeague 8
concept:teamPlaysInLeague_inv 9
EOL

# Create kb_env_rl.txt (copy of graph.txt)
cp "$REPO_ROOT/NELL-995/tasks/$relation/graph.txt" "$REPO_ROOT/NELL-995/kb_env_rl.txt"

# Create entity and relation embeddings
# This is a simple approach - random vectors with correct dimensions
python3 -c "
import numpy as np

# Create entity embeddings (100-dim as per the TensorFlow implementation)
entity_count = 8
entity_dim = 100
np.random.seed(42)  # For reproducibility
entity_embeddings = np.random.normal(0, 0.1, (entity_count, entity_dim))
np.savetxt('$REPO_ROOT/NELL-995/entity2vec.bern', entity_embeddings)

# Create relation embeddings (100-dim as per the TensorFlow implementation)
relation_count = 10
relation_dim = 100
relation_embeddings = np.random.normal(0, 0.1, (relation_count, relation_dim))
np.savetxt('$REPO_ROOT/NELL-995/relation2vec.bern', relation_embeddings)
"

echo "Simplified benchmark dataset created at $REPO_ROOT/NELL-995/tasks/$relation/"
echo "You can now run the benchmark with:"
echo "./run_complete_benchmark.sh $relation"