🎥 **Project Demo Video:** [YouTube Link Here]

*Video includes:*
- Overview of the AI Akinator system
- Demonstration of multi-agent architecture
- Live gameplay showing question strategy
- Explanation of information gain algorithm
- LangSmith learning dashboard

---

## Plan

### ✅ Phase 1 - Project Setup [DONE]

Got the basic structure set up:
- Made all the folders (src/, knowledge_base/, tests/, etc.)
- Created virtual environment
- Wrote config.py with Pydantic for managing settings (API keys and stuff)
- Set up the knowledge base folder structure (persons, characters, animals, places, objects)
- Made a setup script so others can run it easily
- Basic README and docs

**What I learned**: How to structure a proper Python project, not just throw everything in one file

**Commits**: "Initial commit", "Phase 1 Complete: Project setup and capstone README"

### Phase 2 - State Management & Data Models [TODO]

Build the core data structures:
- Make the `AkinatorGameState` TypedDict (holds questions asked, remaining candidates, probabilities, etc.)
- Create `Entity` model for people/characters (with Pydantic validation)
- Write functions to initialize and update game state
- Add tests for the models

This is where I define how data flows through the system.

---

### Phase 3 - Knowledge Base & Entity Database [TODO]

Build the brain - where all the entities live:
- Create `EntityDatabaseAgent` that loads JSON files
- Add search/filter functions (find all scientists, filter by attributes, etc.)
- Populate with 100+ entities (Einstein, Harry Potter, etc.)
- Write tests to make sure search works

**RAG part**: Agent retrieves entity info from files instead of making stuff up.

---

### Phase 4 - Question Strategy Agent [TODO]

The smart part - picking good questions:
- Implement entropy calculation (measure how uncertain we are)
- Build information gain algorithm (which question reduces uncertainty most)
- Generate question pool from entity attributes
- Convert attributes to natural questions ("is_alive" → "Is this person still alive?")
- Test that it actually picks good questions

**Why this matters**: Without this, it would just ask random questions. Information theory makes it smart.

---

### Phase 5 - Answer Analysis & Probability Updates [TODO]

Process user answers and update probabilities:
- Build `AnswerAnalysisAgent` 
- Implement Bayesian probability updates (P(entity|answer) = ...)
- Handle fuzzy answers (yes=1.0, maybe=0.5, no=0.0)
- Update probabilities for all candidates
- Test that it converges to right answer

**Math part**: Uses Bayes theorem to update beliefs based on evidence.

---

### Phase 6 - Game Coordinator & Main Loop [TODO]

Put it all together:
- Create main `AkinatorGame` class that runs everything
- Build game loop (ask question → get answer → update → repeat)
- Add logic for when to guess vs keep asking
- Make CLI so users can actually play
- Handle win/lose scenarios
- End-to-end testing

**This is where it becomes a real game people can play.**

---

### Phase 7 - LangSmith Integration & Learning [TODO]

Make it learn from mistakes:
- Set up LangSmith tracing
- Add @traceable to all agent functions
- Build `LearningAgent` that analyzes games
- Track which questions work best
- Record failures and missing knowledge
- Let users teach it new entities

**Continuous improvement**: Gets smarter every game.

---

### Phase 8 - Polish the Interface [TODO]

Make it look good:
- Use Rich library for colored terminal output
- Add progress bars and formatting
- Show statistics (win rate, avg questions, etc.)
- Better error messages
- Game history

**Because ugly CLIs are sad.**

---

### Phase 9 - Testing Everything [TODO]

Make sure it actually works:
- Write unit tests for every agent
- Integration tests for full games
- Try to break it with edge cases
- Aim for 80%+ code coverage
- Performance testing

**Professional code = tested code.**

---

### Phase 10 - Final Documentation & Video [TODO]

Finish strong:
- Clean up code comments
- Write final documentation
- Make architecture diagrams
- Record the demo video
- Update this README with results
- Push everything to GitHub

**The presentation matters.**

---

## How It Works (Technical Stuff)

The multi-agent system looks like this:

```
Main Coordinator
    ├── Question Agent (picks best question using entropy)
    ├── Analysis Agent (updates probabilities with Bayes theorem)  
    ├── Database Agent (searches knowledge base)
    └── Learning Agent (records to LangSmith)
```

**Game Flow**:
1. Start → Initialize candidates
2. Loop: Pick question → User answers → Update probabilities
3. If confident (>85%) → Make guess
4. If correct → Win! If wrong → Keep asking
5. Max 20 questions

**State**: Tracks everything in `AkinatorGameState` - questions asked, remaining candidates, probabilities, etc.

---

## Tech Stack

**LangGraph Stuff**:
- LangGraph for multi-agent coordination
- LangChain for LLM integration  
- LangSmith for debugging and learning
- Claude Sonnet 4 as the LLM

**Python Stuff**:
- Pydantic for data validation
- pytest for testing
- Rich for pretty terminal output

---

## Setup

```bash
# Clone and navigate
cd akinator-ai

# Easy way - run the setup script
python setup_project.py

# OR do it manually:
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
```

**Need API keys from**:
- Anthropic (for Claude): console.anthropic.com
- LangSmith: smith.langchain.com

Both have free tiers.

---

## Running It

```bash
cd akinator-ai
venv\Scripts\activate
python main.py  # after Phase 6 is done
```

---

## What I'm Learning

**From this project**:
- How to actually build multi-agent systems (not just read docs about them)
- Information theory isn't just theoretical - it makes real difference in question selection
- State management gets complicated fast with multiple agents
- LangSmith is really useful for debugging agent behavior  
- Bayesian updates are cool when you actually implement them

**Challenges so far**:
- Figuring out proper project structure (Phase 1 took a while)
- Understanding how to coordinate multiple agents
- Balancing between over-engineering and making it work

---

## Conclusion

**What I wanted to do**:
- Build a working Akinator clone using all MAT496 concepts
- Make it actually smart (not just random questions)
- Get it to learn from games
- Have clean, tested code

**Status**: Phase 1 done, 9 more to go

**Am I satisfied so far?** Yeah, pretty satisfied. The foundation is solid - proper structure, good setup, all the concepts mapped out. The hard part (implementing the actual algorithms) is still ahead, but I'm confident the architecture will work.

The plan is clear, I understand the math behind it (entropy, Bayes theorem), and I know which LangGraph tools to use where. Now it's just a matter of implementing each phase carefully.

**Biggest learning**: Don't jump straight into coding. Spending time on Phase 1 (structure, planning, understanding) makes the rest way easier.

---

## Next Steps

- [ ] Phase 2: Build the state management system
- [ ] Phase 3: Populate knowledge base with entities  
- [ ] Phase 4: im plement information gain algorithm
- [ ] Record the demo video once it's playable

One phase at a time, one commit per phase.

---

*Built for MAT496 Capstone Project - Dec 2024*
