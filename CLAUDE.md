# Claude Instructions for GraphReasoningRL Project

## CRITICAL RULES

### NEVER USE SIMULATIONS OR FAKE RESULTS
- **NEVER EVER** use 'simulations' to produce results
- **NEVER** create fake or simulated data to make comparisons appear to work
- **ALWAYS** fix the actual code issues instead of simulating results
- If code doesn't work, fix it properly - don't work around it with simulations
- Real implementations only - no shortcuts, no fake data, no simulations

### Implementation Comparison Requirements
- Both implementations must actually run and produce real results
- Fix syntax errors, dependency issues, and compatibility problems properly
- Only compare real results from real code execution
- Document limitations honestly if real comparison isn't possible

### Code Quality Standards
- Fix Python 2 to Python 3 syntax issues properly
- Handle dependency conflicts by creating proper environments
- Test all scripts thoroughly before claiming they work
- Verify that all comparison tools actually function as intended

## Project Structure
- `deeppath/` - DeepPath implementation (PyTorch)
- `minerva/` - MINERVA implementation (PyTorch + TensorFlow)
- Both should have working `comparison/` directories with real comparison tools

## Testing Requirements
- All scripts in `comparison/` directories must work
- All implementations must produce real results
- Comparisons must be between actual running code
- No simulations, mock data, or fake results allowed