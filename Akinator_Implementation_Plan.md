# AI Akinator - 10 Phase Implementation Plan
## Complete Development Roadmap

---

## 📋 Project Overview

**Timeline:** 8-10 weeks  
**Tech Stack:** LangGraph, LangSmith, Python, LangChain  
**Goal:** Production-ready AI Akinator with learning capabilities

---

## Phase 1: Project Setup & Foundation (Week 1 - Days 1-3)

### Objectives
- Set up development environment
- Initialize project structure
- Configure LangSmith & API keys
- Create basic knowledge base schema

### Deliverables
```
akinator-ai/
├── .env                          # API keys
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── agents/
│   ├── models/
│   ├── utils/
│   └── config.py
├── knowledge_base/
│   ├── persons/
│   ├── characters/
│   └── schema.json
├── tests/
└── scripts/
```

### Tasks
- [x] Create virtual environment
- [ ] Install dependencies: `langgraph`, `langchain`, `langchain-anthropic`, `langsmith`
- [ ] Set up `.env` with ANTHROPIC_API_KEY, LANGCHAIN_API_KEY
- [ ] Create project folder structure
- [ ] Initialize git repository
- [ ] Write basic README

### Code Snippets

**requirements.txt**
```txt
langgraph>=0.2.0
langchain>=0.3.0
langchain-anthropic>=0.2.0
langsmith>=0.1.0
python-dotenv>=1.0.0
pydantic>=2.0.0
```

**config.py**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str
    langchain_api_key: str
    langchain_project: str = "akinator-ai"
    max_questions: int = 20
    confidence_threshold: float = 0.85
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### Success Criteria
✅ All dependencies installed  
✅ Project structure created  
✅ Environment variables configured  
✅ Can import langchain/langgraph successfully

---

## Phase 2: State Management & Data Models (Week 1 - Days 4-7)

### Objectives
- Define game state schema
- Create entity data models
- Implement file-based knowledge storage
- Build state management utilities

### Deliverables

**models/state.py**
```python
from typing import TypedDict, Dict, List, Literal
from datetime import datetime

class AkinatorGameState(TypedDict):
    session_id: str
    category: str
    
    # Question tracking
    messages: List[Dict]
    questions_asked: List[Dict]
    current_question: str
    total_questions: int
    max_questions: int
    
    # Entity management
    candidate_entities: Dict[str, float]  # {entity_id: probability}
    eliminated_entities: List[str]
    top_guess: str
    
    # File system (your pattern)
    files: Dict[str, str]
    
    # TODO tracking
    todos: List[Dict]
    
    # Game state
    confidence_threshold: float
    game_outcome: Literal["won", "lost", "ongoing"]
    
    # Metadata
    started_at: str
    ended_at: str
```

**models/entity.py**
```python
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class Entity(BaseModel):
    id: str
    name: str
    category: str
    attributes: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "person_001",
                "name": "Albert Einstein",
                "category": "person",
                "attributes": {
                    "is_real": True,
                    "is_alive": False,
                    "gender": "male",
                    "occupation": "scientist"
                }
            }
        }
```

### Tasks
- [ ] Define `AkinatorGameState` TypedDict
- [ ] Create `Entity` Pydantic model
- [ ] Implement JSON serialization/deserialization
- [ ] Build state initialization function
- [ ] Create state update utilities
- [ ] Write unit tests for models

### Key Functions

```python
def initialize_game_state(category: str) -> AkinatorGameState:
    """Initialize fresh game state"""
    return AkinatorGameState(
        session_id=str(uuid.uuid4()),
        category=category,
        messages=[],
        questions_asked=[],
        current_question="",
        total_questions=0,
        max_questions=20,
        candidate_entities={},
        eliminated_entities=[],
        top_guess="",
        files={},
        todos=[
            {"content": "Ask questions", "status": "pending"},
            {"content": "Make guess", "status": "pending"}
        ],
        confidence_threshold=0.85,
        game_outcome="ongoing",
        started_at=datetime.now().isoformat(),
        ended_at=""
    )
```

### Success Criteria
✅ All models defined with type hints  
✅ State can be serialized to JSON  
✅ Unit tests passing  
✅ Example state created successfully

---

## Phase 3: Knowledge Base & Entity Manager (Week 2)

### Objectives
- Build entity database agent
- Create knowledge base loader
- Implement entity search/filter
- Populate initial knowledge base with 50+ entities

### Deliverables

**agents/entity_database_agent.py**
```python
import json
import os
from typing import Dict, List, Optional
from models.entity import Entity

class EntityDatabaseAgent:
    def __init__(self, kb_path: str = "./knowledge_base"):
        self.kb_path = kb_path
        self.entities: Dict[str, Entity] = {}
        self.load_all_entities()
    
    def load_all_entities(self) -> None:
        """Load all entities from JSON files"""
        categories = ['persons', 'characters', 'animals', 'places', 'objects']
        
        for category in categories:
            category_path = os.path.join(self.kb_path, category)
            if os.path.exists(category_path):
                for filename in os.listdir(category_path):
                    if filename.endswith('.json'):
                        filepath = os.path.join(category_path, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            entity = Entity(**data)
                            self.entities[entity.id] = entity
        
        print(f"📚 Loaded {len(self.entities)} entities")
    
    def get_entities_by_category(self, category: str) -> Dict[str, Entity]:
        """Get all entities in a category"""
        return {
            eid: e for eid, e in self.entities.items() 
            if e.category == category
        }
    
    def add_entity(self, entity: Entity) -> str:
        """Add new entity to knowledge base"""
        self.entities[entity.id] = entity
        
        # Save to file
        category_path = os.path.join(self.kb_path, f"{entity.category}s")
        os.makedirs(category_path, exist_ok=True)
        
        safe_name = entity.name.lower().replace(' ', '_')
        filepath = os.path.join(category_path, f"{safe_name}.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(entity.model_dump(), f, indent=2)
        
        return entity.id
```

### Tasks
- [ ] Create `EntityDatabaseAgent` class
- [ ] Implement load/save operations
- [ ] Add entity search by name
- [ ] Build filter by attributes
- [ ] Create entity addition workflow
- [ ] Write knowledge base population script

### Knowledge Base Population

**scripts/populate_kb.py**
```python
from models.entity import Entity
from agents.entity_database_agent import EntityDatabaseAgent

def create_sample_entities():
    entities = [
        Entity(
            id="person_001",
            name="Albert Einstein",
            category="person",
            attributes={
                "is_real": True,
                "is_alive": False,
                "gender": "male",
                "occupation": "scientist",
                "field": "physics",
                "nationality": "german",
                "has_nobel_prize": True,
                "famous_for": "relativity"
            }
        ),
        # Add 49 more...
    ]
    
    db = EntityDatabaseAgent()
    for entity in entities:
        db.add_entity(entity)
    
    print(f"✅ Added {len(entities)} entities")

if __name__ == "__main__":
    create_sample_entities()
```

### Success Criteria
✅ 50+ entities in knowledge base  
✅ Entities load successfully  
✅ Can add new entities programmatically  
✅ Search and filter working

---

## Phase 4: Question Strategy Agent (Week 3)

### Objectives
- Implement information gain algorithm
- Build question generation system
- Create attribute extraction logic
- Test question effectiveness

### Core Algorithm

**agents/question_strategy_agent.py**
```python
import math
from typing import Dict, List, Tuple

class QuestionStrategyAgent:
    def calculate_entropy(self, probs: Dict[str, float]) -> float:
        """Calculate Shannon entropy"""
        total = sum(probs.values())
        entropy = 0.0
        
        for p in probs.values():
            if p > 0:
                p_norm = p / total
                entropy -= p_norm * math.log2(p_norm)
        
        return entropy
    
    def calculate_information_gain(
        self, 
        question: str,
        candidates: Dict[str, float],
        entities: Dict
    ) -> float:
        """Calculate expected information gain"""
        current_entropy = self.calculate_entropy(candidates)
        
        # Simulate yes/no answers
        yes_entities = self._filter_by_answer(question, "yes", candidates, entities)
        no_entities = self._filter_by_answer(question, "no", candidates, entities)
        
        # Calculate probabilities
        total = len(candidates)
        p_yes = len(yes_entities) / total if total > 0 else 0
        p_no = len(no_entities) / total if total > 0 else 0
        
        # Expected entropy
        entropy_yes = self.calculate_entropy(yes_entities) if yes_entities else 0
        entropy_no = self.calculate_entropy(no_entities) if no_entities else 0
        
        expected_entropy = p_yes * entropy_yes + p_no * entropy_no
        
        return current_entropy - expected_entropy
    
    def select_best_question(
        self,
        candidates: Dict[str, float],
        asked_questions: List[str],
        entities: Dict
    ) -> str:
        """Select question with highest information gain"""
        potential_questions = self.generate_questions(candidates, entities)
        available = [q for q in potential_questions if q not in asked_questions]
        
        if not available:
            return "Is this entity well-known?"
        
        # Calculate gains
        gains = {
            q: self.calculate_information_gain(q, candidates, entities)
            for q in available
        }
        
        return max(gains, key=gains.get)
    
    def generate_questions(self, candidates: Dict, entities: Dict) -> List[str]:
        """Generate question pool from top candidates"""
        top_entities = sorted(
            candidates.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        all_attributes = set()
        for eid, _ in top_entities:
            if eid in entities:
                all_attributes.update(entities[eid].attributes.keys())
        
        questions = []
        for attr in all_attributes:
            question = self._formulate_question(attr)
            if question:
                questions.append(question)
        
        return questions
    
    def _formulate_question(self, attribute: str) -> str:
        """Convert attribute to natural question"""
        question_templates = {
            "is_real": "Is this a real person/thing (not fictional)?",
            "is_alive": "Is this person still alive?",
            "gender": "Is this person male?",
            "occupation": "Is this person's occupation related to science?",
            # Add more mappings...
        }
        return question_templates.get(attribute, f"Does this have attribute: {attribute}?")
```

### Tasks
- [ ] Implement entropy calculation
- [ ] Build information gain calculator
- [ ] Create question generator
- [ ] Add question templates
- [ ] Test with sample data
- [ ] Optimize for speed

### Success Criteria
✅ Entropy calculation correct  
✅ Information gain maximized  
✅ Questions generated naturally  
✅ Performance acceptable (<100ms per question)

---

## Phase 5: Answer Analysis & Probability Updates (Week 4)

### Objectives
- Implement Bayesian probability updates
- Build answer interpretation system
- Handle fuzzy answers (maybe, probably)
- Test probability convergence

### Implementation

**agents/answer_analysis_agent.py**
```python
from typing import Dict, Tuple

class AnswerAnalysisAgent:
    def __init__(self):
        self.answer_weights = {
            'yes': 1.0,
            'probably': 0.8,
            'maybe': 0.5,
            "don't know": 0.5,
            'probably not': 0.2,
            'no': 0.0
        }
    
    def update_probabilities(
        self,
        question: str,
        answer: str,
        candidates: Dict[str, float],
        entities: Dict
    ) -> Dict[str, float]:
        """Bayesian probability update"""
        attribute, expected_value = self._parse_question(question)
        answer_weight = self.answer_weights.get(answer, 0.5)
        
        updated = {}
        for eid, prob in candidates.items():
            entity = entities[eid]
            entity_value = entity.attributes.get(attribute)
            
            if entity_value is None:
                # Unknown - keep probability
                updated[eid] = prob
            else:
                # Calculate likelihood
                if entity_value == expected_value:
                    likelihood = answer_weight
                else:
                    likelihood = 1.0 - answer_weight
                
                # Bayesian update
                updated[eid] = prob * likelihood
        
        # Normalize
        total = sum(updated.values())
        if total > 0:
            updated = {k: v/total for k, v in updated.items()}
        
        return updated
    
    def _parse_question(self, question: str) -> Tuple[str, any]:
        """Extract attribute and expected value from question"""
        q_lower = question.lower()
        
        # Pattern matching
        if "real" in q_lower and "fictional" in q_lower:
            return ("is_real", True)
        elif "alive" in q_lower:
            return ("is_alive", True)
        elif "male" in q_lower:
            return ("gender", "male")
        # Add more patterns...
        
        return ("unknown", None)
```

### Tasks
- [ ] Implement Bayesian update formula
- [ ] Create answer weight system
- [ ] Build question parser
- [ ] Handle edge cases
- [ ] Test convergence speed
- [ ] Validate accuracy

### Success Criteria
✅ Probabilities sum to 1.0  
✅ Correct entities get higher probabilities  
✅ Converges to answer within 20 questions  
✅ Handles ambiguous answers

---

## Phase 6: Game Coordinator & Main Loop (Week 5)

### Objectives
- Build main game coordinator agent
- Implement game loop
- Add decision-making logic
- Test end-to-end gameplay

### Implementation

**akinator_game.py**
```python
from agents.entity_database_agent import EntityDatabaseAgent
from agents.question_strategy_agent import QuestionStrategyAgent
from agents.answer_analysis_agent import AnswerAnalysisAgent
from models.state import AkinatorGameState, initialize_game_state

class AkinatorGame:
    def __init__(self):
        self.db_agent = EntityDatabaseAgent()
        self.question_agent = QuestionStrategyAgent()
        self.analysis_agent = AnswerAnalysisAgent()
        self.state = None
    
    def start_game(self, category: str = "person"):
        """Initialize and start game"""
        self.state = initialize_game_state(category)
        
        # Load entities
        entities = self.db_agent.get_entities_by_category(category)
        
        # Initialize probabilities
        n = len(entities)
        initial_prob = 1.0 / n if n > 0 else 0
        self.state["candidate_entities"] = {
            eid: initial_prob for eid in entities.keys()
        }
        
        print(f"🧞 Think of a {category}!")
        self.game_loop()
    
    def game_loop(self):
        """Main game loop"""
        while self.should_continue():
            # Check if should guess
            if self.should_make_guess():
                if self.make_guess():
                    return  # Won
            
            # Ask question
            question = self.question_agent.select_best_question(
                self.state["candidate_entities"],
                [q["question"] for q in self.state["questions_asked"]],
                self.db_agent.entities
            )
            
            answer = self.ask_question(question)
            self.update_state(question, answer)
            
            self.state["total_questions"] += 1
        
        self.handle_gave_up()
    
    def should_continue(self) -> bool:
        """Check if game should continue"""
        return (
            self.state["total_questions"] < self.state["max_questions"]
            and self.state["game_outcome"] == "ongoing"
            and len(self.state["candidate_entities"]) > 0
        )
    
    def should_make_guess(self) -> bool:
        """Determine if confident enough to guess"""
        if not self.state["candidate_entities"]:
            return False
        
        max_prob = max(self.state["candidate_entities"].values())
        return (
            max_prob >= self.state["confidence_threshold"]
            or len(self.state["candidate_entities"]) <= 3
        )
    
    def ask_question(self, question: str) -> str:
        """Display question and get answer"""
        print(f"\n❓ Q{self.state['total_questions'] + 1}: {question}")
        print("   [y/n/m/dk/p/pn]")
        
        answer = input("   > ").strip().lower()
        
        answer_map = {
            'y': 'yes', 'n': 'no', 'm': 'maybe',
            'dk': "don't know", 'p': 'probably', 'pn': 'probably not'
        }
        
        return answer_map.get(answer, 'maybe')
    
    def update_state(self, question: str, answer: str):
        """Update game state after answer"""
        self.state["questions_asked"].append({
            "question": question,
            "answer": answer,
            "candidates_before": len(self.state["candidate_entities"])
        })
        
        # Update probabilities
        updated = self.analysis_agent.update_probabilities(
            question, answer,
            self.state["candidate_entities"],
            self.db_agent.entities
        )
        
        # Filter low probabilities
        self.state["candidate_entities"] = {
            k: v for k, v in updated.items() if v >= 0.001
        }
        
        print(f"   📊 Candidates: {len(self.state['candidate_entities'])}")
    
    def make_guess(self) -> bool:
        """Make final guess"""
        top_id = max(
            self.state["candidate_entities"],
            key=self.state["candidate_entities"].get
        )
        entity = self.db_agent.entities[top_id]
        
        print(f"\n🎯 Is it... {entity.name}?")
        answer = input("   [y/n] > ").strip().lower()
        
        if answer == 'y':
            print(f"🎉 Won in {self.state['total_questions']} questions!")
            self.state["game_outcome"] = "won"
            return True
        else:
            del self.state["candidate_entities"][top_id]
            return False
    
    def handle_gave_up(self):
        """Handle game over"""
        print("😔 I give up!")
        self.state["game_outcome"] = "lost"
```

### Tasks
- [ ] Build `AkinatorGame` class
- [ ] Implement game loop
- [ ] Add guess logic
- [ ] Handle user input
- [ ] Test complete games
- [ ] Fix bugs

### Success Criteria
✅ Can play complete game  
✅ Guesses correctly within 20 questions  
✅ Handles edge cases  
✅ User experience smooth

---

## Phase 7: LangSmith Integration & Learning (Week 6)

### Objectives
- Integrate LangSmith tracing
- Build learning agent
- Record game outcomes
- Implement feedback loops

### Implementation

**agents/learning_agent.py**
```python
from langsmith import Client, traceable
from typing import List, Dict

class LearningAgent:
    def __init__(self):
        self.client = Client()
    
    @traceable(name="record_game", tags=["game"])
    def record_game_outcome(
        self,
        state: Dict,
        entity_id: str,
        won: bool
    ):
        """Record game to LangSmith"""
        self.client.create_feedback(
            run_id=None,
            key="game_outcome",
            score=1.0 if won else 0.0,
            value={
                "entity_id": entity_id,
                "questions_used": state["total_questions"],
                "won": won
            }
        )
    
    @traceable(name="analyze_questions", tags=["learning"])
    def analyze_question_effectiveness(
        self,
        questions: List[Dict]
    ):
        """Analyze which questions were most effective"""
        for i, q in enumerate(questions):
            if i + 1 < len(questions):
                reduction = (
                    q["candidates_before"] - 
                    questions[i+1]["candidates_before"]
                )
                effectiveness = reduction / q["candidates_before"]
                
                self.client.create_feedback(
                    run_id=None,
                    key="question_effectiveness",
                    score=effectiveness,
                    value={"question": q["question"]}
                )
```

### Tasks
- [ ] Set up LangSmith project
- [ ] Add `@traceable` decorators
- [ ] Record all games
- [ ] Track question effectiveness
- [ ] Build analytics queries
- [ ] Create dashboards

### Success Criteria
✅ All games logged to LangSmith  ✅ Can view game history  
✅ Analytics dashboard working  
✅ Feedback loops active

---

## Phase 8: CLI & User Interface (Week 7)

### Objectives
- Build polished CLI interface
- Add colors and formatting
- Improve user experience
- Add game statistics

### Implementation

**main.py**
```python
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from akinator_game import AkinatorGame

console = Console()

def main():
    console.print(Panel.fit(
        "[bold blue]🧞 AI Akinator[/bold blue]\n"
        "Think of someone, and I'll guess who!",
        border_style="blue"
    ))
    
    categories = {
        "1": "person",
        "2": "character",
        "3": "animal"
    }
    
    console.print("\n[cyan]Choose category:[/cyan]")
    for k, v in categories.items():
        console.print(f"  {k}. {v.capitalize()}")
    
    choice = Prompt.ask("Your choice", choices=list(categories.keys()))
    category = categories[choice]
    
    game = AkinatorGame()
    game.start_game(category)

if __name__ == "__main__":
    main()
```

### Tasks
- [ ] Install `rich` library
- [ ] Add colored output
- [ ] Format questions nicely
- [ ] Show progress bar
- [ ] Add statistics display
- [ ] Polish UX

---

## Phase 9: Web Interface (Optional - Week 8)

### Objectives
- Build Streamlit web UI
- Add interactive elements
- Deploy locally/cloud

### Quick Implementation

**app.py**
```python
import streamlit as st
from akinator_game import AkinatorGame

st.title("🧞 AI Akinator")

if 'game' not in st.session_state:
    category = st.selectbox("Category", ["person", "character", "animal"])
    if st.button("Start Game"):
        st.session_state.game = AkinatorGame()
        st.session_state.game.start_game(category)

if 'game' in st.session_state:
    game = st.session_state.game
    
    st.write(f"### Question {game.state['total_questions'] + 1}")
    st.write(game.state['current_question'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Yes"):
            # Handle answer
            pass
```

---

## Phase 10: Testing, Optimization & Deployment (Week 9-10)

### Objectives
- Write comprehensive tests
- Performance optimization
- Documentation
- Deployment

### Testing Checklist

```python
# tests/test_game.py
import pytest
from akinator_game import AkinatorGame

def test_game_initialization():
    game = AkinatorGame()
    assert game is not None

def test_probability_updates():
    # Test Bayesian updates
    pass

def test_question_selection():
    # Test information gain
    pass

def test_complete_game():
    # Simulate full game
    pass
```

### Tasks
- [ ] Unit tests (80%+ coverage)
- [ ] Integration tests
- [ ] Performance profiling
- [ ] Optimization
- [ ] Documentation
- [ ] Docker container
- [ ] Deploy to cloud

---

## 📊 Success Metrics

| Phase | Success Rate Target | Avg Questions | Response Time |
|-------|-------------------|---------------|---------------|
| 1-3   | N/A (Foundation)  | N/A           | N/A           |
| 4-6   | 60%+              | <25           | <2s           |
| 7-8   | 75%+              | <20           | <1s           |
| 9-10  | 85%+              | <18           | <500ms        |

---

## 🎯 Final Deliverables

- ✅ Working CLI game
- ✅ 200+ entity knowledge base
- ✅ LangSmith integration
- ✅ Test suite (80%+ coverage)
- ✅ Documentation
- ✅ (Optional) Web interface
- ✅ (Optional) Deployed version

---

**Start with Phase 1 and work sequentially. Each phase builds on the previous one!** 🚀
