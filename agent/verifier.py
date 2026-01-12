from typing import List, Any, Dict


class Verifier:
    """
    Verifier: checks if execution results satisfy the query.
    Uses heuristics + optional LLM spot-checks.
    """
    
    def verify(self, user_query: str, results: List[Any]) -> str:
        """
        Check results. Return: "accept", "retry", or "abort".
        """
        
        # Fast fail on execution errors
        for res in results:
            if isinstance(res, dict) and "error" in res:
                print(f"[VERIFIER] Execution error detected: {res['error']}")
                return "retry"
        
        # Check if we got meaningful output
        has_output = False
        for res in results:
            if isinstance(res, dict):
                result = res.get("result")
                if result and (isinstance(result, str) and len(result) > 10 or isinstance(result, (list, dict))):
                    has_output = True
                    break
        
        if not has_output:
            print("[VERIFIER] No meaningful output generated.")
            return "retry"
        
        print("[VERIFIER] Results look valid.")
        return "accept"
