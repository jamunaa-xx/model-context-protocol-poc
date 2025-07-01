import os
import json

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="Resource", host = "0.0.0.0", port = 5757)

@mcp.tool()
def get_knowledge_base()->str:
    """
    Retrieve the entire knowledge base as formatted string.
    
    Returns:
        Formatted string containing all the Q&A pairs from the knowledge base.
    """
    
    try:
        kb_file = os.path.join(os.path.dirname(__file__), "data", "knowledge_base.json")
        
        with open(kb_file, "r") as f:
            knowledge_base = json.load(f)
            
        kb_text = "Here is the retrieved knowledge base:\n\n"
        
        if isinstance(knowledge_base, list):
            for i, item in enumerate(knowledge_base, 1):
                if isinstance(item, dict):
                    question = item.get("question", "No question provided")
                    answer = item.get("answer", "No answer")
                else:
                    question = f"Item {i}"
                    answer = str(i)
                kb_text += f"Q{i}: {question}\nA{i}: {answer}\n\n"
                
        else:
            kb_text += f"Knowledge base content: {json.dumps(knowledge_base, indent=2)}\n\n"
            
        return kb_text
    except FileNotFoundError:
        return "Error: Knowledge base file not found."
    except json.JSONDecodeError:
        return "Error: Invalid JSON format in knowledge base file."
    except Exception as e:
        return f"Error: {str(e)}"
    
    
if __name__ == "__main__":
    mcp.run(transport="stdio")
            