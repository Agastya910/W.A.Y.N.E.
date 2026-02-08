---
description: How to run Python commands in RepoPilot using the virtual environment
---

# Running Python in RepoPilot

RepoPilot uses a Python virtual environment located at `/home/agastya/RepoPilot/venv`.

## Running a Python script
// turbo-all

1. Activate venv and run the script:
```bash
wsl -e bash -c "cd /home/agastya/RepoPilot && source venv/bin/activate && python <script.py>"
```

## Running the CLI

1. Run the main CLI:
```bash
wsl -e bash -c "cd /home/agastya/RepoPilot && source venv/bin/activate && python cli.py ."
```

## Running a quick Python command

1. Run inline Python:
```bash
wsl -e bash -c "cd /home/agastya/RepoPilot && source venv/bin/activate && python -c '<code>'"
```

## Compiling/syntax check

1. Check Python files compile:
```bash
wsl -e bash -c "cd /home/agastya/RepoPilot && source venv/bin/activate && python -m py_compile <file1.py> <file2.py>"
```

## Notes

- Always use `source venv/bin/activate` before running Python
- Use `python` (not `python3`) after activating venv
- Working directory should be `/home/agastya/RepoPilot`
