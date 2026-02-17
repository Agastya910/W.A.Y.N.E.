import argparse
import os
import sys
from agent.planner import Planner
from agent.executor import Executor
from agent.verifier import Verifier
from agent.chat_history import ChatHistory


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
        description="WAYNE - Webless Autonomous Neural Engine"
    )
    parser.add_argument("repo_path", help="Path to the repository to analyze.")
    args = parser.parse_args()

    repo_path = os.path.abspath(args.repo_path)
    
    if not os.path.exists(repo_path):
        print(f"[ERROR] Repository path does not exist: {repo_path}")
        sys.exit(1)

    print(f"""
╔══════════════════════════════════════════════════════╗
║    W.A.Y.N.E. - Webless Autonomous Neural Engine     ║
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
        history = ChatHistory(repo_path)
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
            edit_info = None
            if executor.has_pending_edit():
                edit_info = executor.get_pending_edit_info()
                confirm = input("Apply this edit? [y/n]: ").strip().lower()
                if confirm == 'y':
                    apply_result = executor._apply_edit_tool(confirm=True)
                    if apply_result.get("success"):
                        print(f"\n{apply_result.get('message')}")
                    else:
                        print(f"\n❌ {apply_result.get('message')}")
                else:
                    executor._apply_edit_tool(confirm=False)
                    edit_info = None # Don't log cancelled edits as "edited"
                    print("\n❌ Edit cancelled.")
            
            # Log turn to history
            last_action = results[-1].get("tool", "unknown") if results else "none"
            history.add_turn(query, last_action, edit_info)
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"[ERROR] {e}\n")


if __name__ == "__main__":
    main()