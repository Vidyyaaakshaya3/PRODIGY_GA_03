# 📖 Text Generation with Markov Chains

## 📌 Internship Task - Prodigy InfoTech (Generative AI Intern)

As part of my internship at *Prodigy InfoTech, I implemented a **Markov Chain-based text generation system*, focused on story creation. This project deepened my understanding of probabilistic models, natural language generation, and creative applications of AI in storytelling.

---

## 🛠 Implementation Overview

✅ *Markov Chain-Based Text Model*  
- Built a custom n-gram (state-size) model using Markov Chains.  
- Tokenized and cleaned training data to preserve sentence structure and track sentence starters.

✅ *Training Process*  
- Trained the model on a diverse corpus of fictional stories ranging from fantasy and sci-fi to detective and magical themes.  
- Constructed transition probabilities between word states for text generation.

✅ *Interactive Story Generator*  
- Created a CLI-based interface allowing users to:
  - Generate random or prompt-based stories
  - Train on new story data
  - Customize output length and model state size

✅ *Prompt-Aware Generation*  
- Included logic to find the best starting state based on keywords from the user’s input prompt.

✅ *Model Statistics and Debug Info*  
- Added functions to display model state counts, top transitions, and generation stats for evaluation.

---

## 🖼 Output Sample

> 📝 Generated short stories based on prompts like "space exploration", "fantasy adventure", and "haunted house".

Example:

Prompt: "fantasy adventure"
Story: The young wizard Harry practiced his spells in the ancient library. Magic filled the air as he cast enchantments and created potions. The spaceship landed on the alien planet with a loud crash. The knight rode his horse through the enchanted forest.


---

## 💡 Features

- 🧠 *Markov Chain Logic*: Predicts next word based on preceding words.
- 🎯 *Prompt Matching*: Context-aware start state based on input prompt.
- 🔁 *Story Flow*: Sentence starters ensure grammatical and meaningful story transitions.
- 🛠 *State Size Control*: Users can modify n-gram size for creativity vs. coherence.

---

## 🔗 References

- [#1 GeeksforGeeks - Markov Chains for Text](https://www.geeksforgeeks.org/markov-chain-text-generation/)  
- [#2 Towards Data Science - Simple Markov Chain Text Generator](https://towardsdatascience.com/markov-chain-text-generation-20b4d0e90d90)

---
