import argparse
import os
import sys
from agent.planner import Planner
from agent.executor import Executor
from agent.verifier import Verifier


def print_result(res: dict, max_chars: int = None):
    """Pretty print a result, with optional truncation."""
    tool = res.get("tool", "unknown")
    
    if "error" in res:
        print(f"❌ {tool}: {res['error']}")
        return
    
    result = res.get("result")
    
    # Skip printing llm_analysis as it's already streamed during planning
    if tool == "llm_analysis":
        return
    
    # Show full result without truncation for important tools
    if tool in ["report"]:
        if isinstance(result, str):
            print(result)
        else:
            import json
            print(json.dumps(result, indent=2))
    else:
        # For other tools, show snippet
        if isinstance(result, str):
            display = result if not max_chars else result[:max_chars]
            print(display)
            if max_chars and len(result) > max_chars:
                print(f"\n... [output truncated, total length: {len(result)} chars]")
        else:
            import json
            output = json.dumps(result, indent=2)
            display = output if not max_chars else output[:max_chars]
            print(display)
            if max_chars and len(output) > max_chars:
                print(f"\n... [output truncated]")


def main():
    parser = argparse.ArgumentParser(
        description="RepoPilot - Offline-First Codebase Intelligence Agent"
    )
    parser.add_argument("repo_path", help="Path to the repository to analyze.")
    args = parser.parse_args()

    repo_path = os.path.abspath(args.repo_path)
    
    if not os.path.exists(repo_path):
        print(f"[ERROR] Repository path does not exist: {repo_path}")
        sys.exit(1)

    print(f"""
╔══════════════════════════════════════════════════════╗
║          🚀 RepoPilot - Local LLM Agent 🚀          ║
║    Offline-first codebase intelligence system        ║
╚══════════════════════════════════════════════════════╝
    """)
    print(f"Repository: {repo_path}")
    print("Building semantic index...\n")
    
    # Initialize agents
    try:
        planner = Planner(repo_path)
        executor = Executor(repo_path)
        verifier = Verifier()
    except Exception as e:
        print(f"[ERROR] Failed to initialize agent: {e}")
        sys.exit(1)
    
    print("[OK] Agent initialized. Ready for queries.\n")
    
    # Interactive loop
    while True:
        try:
            query = input("\nHow can I help you? (or 'exit' to quit) > ").strip()
            
            if not query:
                continue
            
            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            # Plan → Execute → Verify
            print("\n[PLANNING]...")
            plan = planner.create_plan(query)
            
            print("\n[EXECUTING]...")
            results = executor.execute_plan(plan)
            
            print("\n[VERIFYING]...")
            status = verifier.verify(query, results)
            
            # Display results - FULL OUTPUT, NO TRUNCATION
            print("\n" + "="*60)
            for res in results:
                # Special handling for edit_file results
                if res.get("tool") == "edit_file":
                    edit_result = res.get("result", {})
                    if edit_result.get("success"):
                        print("\n[EDIT PREVIEW]")
                        print(f"File: {edit_result.get('file_path')}")
                        print(f"Summary: {edit_result.get('summary')}")
                        print("\n--- Diff ---")
                        print(edit_result.get('diff', '[No diff]'))
                        print("--- End Diff ---\n")
                    else:
                        print(f"❌ Edit failed: {edit_result.get('error', 'Unknown error')}")
                else:
                    print_result(res)
            
            print("="*60)
            print(f"Status: {'✅ ACCEPT' if status == 'accept' else '⚠️  RETRY' if status == 'retry' else '❌ ABORT'}\n")
            
            # Check for pending edit and ask for confirmation
            if executor.has_pending_edit():
                confirm = input("Apply this edit? [y/n]: ").strip().lower()
                if confirm == 'y':
                    apply_result = executor._apply_edit_tool(confirm=True)
                    if apply_result.get("success"):
                        print(f"\n{apply_result.get('message')}")
                    else:
                        print(f"\n❌ {apply_result.get('message')}")
                else:
                    executor._apply_edit_tool(confirm=False)
                    print("\n❌ Edit cancelled.")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"[ERROR] {e}\n")


if __name__ == "__main__":
    main()