## Environment setup

### Virtualenv

Create a virtual environment to isolate dependencies.   
```bash
python3 -m venv ./
``` 

Activate the environment and set up the path.

```bash
source bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Test

Run Tests

```bash
python -m pytest src/
``` 

### Core Logic
The simulation follows a strict cycle:

*   **Rules**: Determines valid moves from the current state (`Rules(State) -> [Action]`).
*   **Agent**: Selects an action from the valid options (`Agent.choose_action([Action]) -> Action`).
*   **State**: Updates itself based on actions (`State.observe(player, Action)`).
*   **Simulator**: Orchestrates the game loop using Rules and Agents to produce a final outcome (`Simulator(Rules, Agents)->simulate(States) -> Result`).

### Implementation Strategy
*   **Core**: Contains the immutable logic of the game (Rules, State definitions, Simulator).
*   **Specific Implementations (e.g., PPO)**: Build upon the Core. A PPO Agent, for example, wraps the Core State with training-specific logic (rewards, policy networks) to learn optimal strategies without modifying the underlying game rules.
