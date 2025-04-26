import numpy as np
from enum import Enum

class ActionType(Enum):
    # Structure actions
    SELECT_SECTION = "select_section"
    
    # Research actions
    INITIAL_QUERY = "initial_query"
    IDENTIFY_GAP = "identify_gap"
    VECTOR_SEARCH = "vector_search"
    WEB_SEARCH = "web_search"
    
    # Content actions
    GENERATE_CONTENT = "generate_content"
    INTEGRATE_FINDINGS = "integrate_findings"
    
    # Navigation actions
    BACK_TO_OUTLINE = "back_to_outline"
    FINALIZE = "finalize"

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state  
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0
        self.available_actions = None
        
    def add_child(self, state, action):
        child = MCTSNode(state, self, action)
        self.children.append(child)
        return child
    
    def update(self, reward):
        self.visits += 1
        self.value += reward
        
    def get_ucb_score(self, exploration_weight=1.0):
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = exploration_weight * np.sqrt(np.log(self.parent.visits) / self.visits) if self.parent else 0
        return exploitation + exploration
    
    def is_fully_expanded(self):
        return self.available_actions is not None and not self.available_actions