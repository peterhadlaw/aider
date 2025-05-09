# Extend `/reasoning-effort` to support specific models

## Background
Currently, aider supports setting the reasoning effort for the main model through:
- Command line flag: `--reasoning-effort`
- Slash command: `/reasoning-effort`

## Proposed Changes
Add support for setting the reasoning effort for the editor model with a separate command line flag:

### New Command Line Argument
Add a new command line argument to allow users to specify the reasoning effort for the editor model:

## Implementation TODOs

### 1. Add command line argument in args.py
- Add `--editor-reasoning-effort` to the "Model settings" group
- Add appropriate help text similar to `--reasoning-effort` 
- Type should be `str` to support both numeric values and text values like "high"/"medium"/"low"

### 2. Update models.py
- Add new method `set_editor_reasoning_effort(self, effort)` to the `Model` class
- Method should check if the editor model exists and is different from the main model
- If so, call the editor model's `set_reasoning_effort` method

### 3. Add slash command in commands.py
- Create a new method `cmd_editor_reasoning_effort(self, args)` in the `Commands` class
- Display current value if no args are provided
- Set the editor model's reasoning effort when args are provided
- Show appropriate warnings if no separate editor model is configured

### 4. Update initialization in main.py
- Add code to check for and apply `args.editor_reasoning_effort` to the model
- Check if the editor model supports reasoning_effort if `args.check_model_accepts_settings` is True
- Display appropriate warnings if the setting is not supported by the editor model
- Add error handling for invalid values

### 5. Testing
- Test with different models that support reasoning_effort
- Test interaction between main model and editor model settings
- Ensure CLI argument validation works properly
- Verify slash command works during an interactive session
